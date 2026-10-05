"""HTTP isolation, pgvector ranking, atomic publication and provider contracts."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError

from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.models.persistence import (
    Chunk,
    DocumentVersion,
    IndexedChunk,
    IndexGeneration,
    KnowledgeBase,
    ProcessingGeneration,
)
from rag_service.services.embeddings import EmbeddingUnavailableError, EndpointEmbedding
from rag_service.services.processing import ProcessingFailedError, process_generation
from rag_service.storage import build_storage

REAL_REQUEST = EndpointEmbedding._request


@pytest.fixture
def library(
    migrated_database: str, tmp_path: Path
) -> Iterator[tuple[TestClient, Any, Settings, str]]:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)
    with TestClient(app) as client:
        kb = client.post("/knowledge-bases", json={"name": "Library"}).json()["id"]
        yield client, app, settings, kb


def complete(app: Any, settings: Settings, uploaded: dict[str, Any]) -> None:
    with app.state.session_factory() as session:
        process_generation(
            session, build_storage(settings), UUID(uploaded["generation_id"]), settings
        )


def search(client: TestClient, kb: str, **options: Any) -> httpx.Response:
    return cast(
        httpx.Response,
        client.post(
            f"/knowledge-bases/{kb}/retrieve", json={"query": "alpha", **options}
        ),
    )


def test_http_isolation_filtering_and_schema(
    library: tuple[TestClient, Any, Settings, str],
) -> None:
    client, app, settings, kb = library
    base = f"/knowledge-bases/{kb}/documents"
    uploaded = client.post(
        base,
        files={"file": ("guide.md", b"# Alpha\nRelevant content")},
        data={
            "metadata": json.dumps(
                {
                    "title": "Guide",
                    "tags": ["reference", "alpha"],
                    "source": "manual",
                    "connector_metadata": {"owner": "team"},
                }
            )
        },
    ).json()
    assert search(client, kb).json()["results"] == []
    complete(app, settings, uploaded)
    response = search(client, kb, result_count=1)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"knowledge_base_id", "results"}
    result = payload["results"][0]
    assert set(result) == {
        "text",
        "score",
        "chunk_id",
        "document_id",
        "version_id",
        "source",
        "metadata",
    }
    assert result["document_id"] == uploaded["document"]["id"]
    assert result["version_id"] == uploaded["version"]["id"]
    assert result["score"] == pytest.approx(1)
    assert result["source"] == {
        "filename": "guide.md",
        "media_type": "text/markdown",
        "checksum": uploaded["version"]["checksum"],
        "location": {
            "section": "Alpha",
            "source_path": "guide.md",
            "line_start": 1,
            "line_end": 2,
        },
    }
    assert result["metadata"] == {
        "title": "Guide",
        "tags": ["reference", "alpha"],
        "source": "manual",
        "connector_metadata": {"owner": "team"},
    }
    for field, value in [("title", "Guide"), ("source", "manual"), ("tag", "alpha")]:
        assert search(
            client, kb, filters=[{"field": field, "operator": "eq", "value": value}]
        ).json()["results"]
        assert search(
            client,
            kb,
            filters=[{"field": field, "operator": "in", "value": ["absent", value]}],
        ).json()["results"]
        assert (
            search(
                client,
                kb,
                filters=[{"field": field, "operator": "eq", "value": "absent"}],
            ).json()["results"]
            == []
        )
    assert (
        search(
            client,
            kb,
            filters=[
                {"field": "title", "operator": "eq", "value": "Guide"},
                {"field": "source", "operator": "eq", "value": "missing"},
            ],
        ).json()["results"]
        == []
    )
    invalid_options: list[dict[str, Any]] = [
        {"query": " "},
        {"result_count": 0},
        {"result_count": 101},
        {"filters": [{"field": "unknown", "operator": "eq", "value": "a"}]},
        {"filters": [{"field": "title", "operator": "in", "value": []}]},
        {"filters": [{"field": "title", "operator": "range", "value": "a"}]},
    ]
    for options in invalid_options:
        assert search(client, kb, **options).status_code == 422
    other = client.post("/knowledge-bases", json={"name": "Other"}).json()["id"]
    assert search(client, other).json()["results"] == []
    assert search(client, other, version_id=result["version_id"]).status_code == 404
    assert search(client, str(uuid4())).status_code == 404
    client.patch(base + "/" + result["document_id"], json={"title": "Renamed"})
    assert (
        search(
            client,
            kb,
            filters=[{"field": "title", "operator": "eq", "value": "Renamed"}],
        ).json()["results"][0]["metadata"]["title"]
        == "Renamed"
    )
    client.delete(base + "/" + result["document_id"])
    assert search(client, kb).json()["results"] == []
    assert search(client, kb, version_id=result["version_id"]).status_code == 404
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(schema["RetrievalResult"]["required"]) == set(result)


def test_latest_active_history_failed_new_source_and_reprocessing(
    library: tuple[TestClient, Any, Settings, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app, settings, kb = library
    base = f"/knowledge-bases/{kb}/documents"
    first = client.post(base, files={"file": ("notes.txt", b"Old content")}).json()
    complete(app, settings, first)
    doc = first["document"]["id"]
    second = client.post(
        base + f"/{doc}/versions", files={"file": ("notes.txt", b"New content")}
    ).json()
    assert (
        search(client, kb).json()["results"][0]["version_id"] == first["version"]["id"]
    )

    def failure(*_args: object, **_kwargs: object) -> list[list[float]]:
        raise EmbeddingUnavailableError

    with monkeypatch.context() as patch:
        patch.setattr(EndpointEmbedding, "_request", failure)
        with pytest.raises(ProcessingFailedError):
            complete(app, settings, second)
    assert (
        search(client, kb).json()["results"][0]["version_id"] == first["version"]["id"]
    )
    complete(app, settings, second)
    assert (
        search(client, kb).json()["results"][0]["version_id"] == second["version"]["id"]
    )
    assert (
        search(client, kb, version_id=first["version"]["id"]).json()["results"][0][
            "text"
        ]
        == "Old content"
    )
    for uploaded in [second, first]:
        version = uploaded["version"]["id"]
        generation = client.post(
            base + f"/{doc}/versions/{version}/reprocess",
            json={},
            headers={"Idempotency-Key": "reindex"},
        ).json()["generation"]
        complete(app, settings, {"generation_id": generation["id"]})
    assert len(search(client, kb).json()["results"]) == 1
    assert (
        search(client, kb).json()["results"][0]["version_id"] == second["version"]["id"]
    )
    restarted = create_app(settings)
    with TestClient(restarted) as restarted_client:
        assert search(restarted_client, kb).json() == search(client, kb).json()


def test_atomic_visibility_replay_and_pinned_identity(
    library: tuple[TestClient, Any, Settings, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app, settings, kb = library
    base = f"/knowledge-bases/{kb}/documents"
    uploaded = client.post(
        base, files={"file": ("notes.txt", b"Durable content")}
    ).json()
    generation_id = UUID(uploaded["generation_id"])
    original = EndpointEmbedding._request

    def observe(
        self: EndpointEmbedding, texts: list[str], *, query: bool
    ) -> list[list[float]]:
        assert search(client, kb).json()["results"] == []
        with app.state.session_factory() as observer:
            assert list(observer.scalars(select(IndexedChunk))) == []
        return original(self, texts, query=query)

    with monkeypatch.context() as patch:
        patch.setattr(EndpointEmbedding, "_request", observe)
        complete(app, settings, uploaded)
    complete(app, settings, uploaded)
    with app.state.session_factory() as session:
        indexed = session.scalars(select(IndexedChunk)).one()
        index = session.scalars(select(IndexGeneration)).one()
        assert (indexed.provider, indexed.model, indexed.dimensions) == (
            "local",
            "Qwen3-Embedding-0.6B",
            1024,
        )
        assert indexed.created_at is not None
        assert indexed.index_generation_id == index.id
        assert indexed.processing_generation_id == generation_id
        assert len(list(session.scalars(select(Chunk)))) == 1
        indexed.model = "different"
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        knowledge_base = session.get_one(KnowledgeBase, UUID(kb))
        knowledge_base.embedding_model = "different"
        session.commit()
    version = uploaded["version"]["id"]
    assert (
        client.post(
            base + f"/{uploaded['document']['id']}/versions/{version}/reprocess",
            json={},
            headers={"Idempotency-Key": "changed-config"},
        ).status_code
        == 409
    )
    assert search(client, kb).json()["results"]


def test_pgvector_ranking_and_dimension_constraint(
    library: tuple[TestClient, Any, Settings, str],
) -> None:
    client, app, settings, kb = library
    first = client.post(
        f"/knowledge-bases/{kb}/documents", files={"file": ("a.txt", b"alpha")}
    ).json()
    second = client.post(
        f"/knowledge-bases/{kb}/documents", files={"file": ("b.txt", b"beta")}
    ).json()
    complete(app, settings, first)
    complete(app, settings, second)
    with app.state.session_factory() as session:
        indexed = session.scalars(
            select(IndexedChunk).where(
                IndexedChunk.processing_generation_id == UUID(second["generation_id"])
            )
        ).one()
        indexed.embedding = [0.0, 1.0] + [0.0] * 1022
        session.commit()
        if session.get_bind().dialect.name == "postgresql":
            assert (
                session.scalar(
                    text("SELECT extname FROM pg_extension WHERE extname='vector'")
                )
                == "vector"
            )
            indexed.embedding = [1.0, 0.0]
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
    results = search(client, kb, result_count=2).json()["results"]
    assert [row["document_id"] for row in results] == [
        first["document"]["id"],
        second["document"]["id"],
    ]
    assert [row["score"] for row in results] == pytest.approx([1, 0])
    assert len(search(client, kb, result_count=1).json()["results"]) == 1


def test_partial_batch_failure_crash_retry_and_previous_generation(
    library: tuple[TestClient, Any, Settings, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app, settings, kb = library
    settings.embedding_batch_size = 1
    with app.state.session_factory() as session:
        knowledge_base = session.get_one(KnowledgeBase, UUID(kb))
        knowledge_base.chunk_size, knowledge_base.chunk_overlap = 4, 1
        session.commit()
    base = f"/knowledge-bases/{kb}/documents"
    uploaded = client.post(
        base, files={"file": ("notes.txt", b"one two three four five six seven eight")}
    ).json()
    complete(app, settings, uploaded)
    before = search(client, kb).json()
    path = base + f"/{uploaded['document']['id']}/versions/{uploaded['version']['id']}"
    generation = client.post(
        path + "/reprocess", json={}, headers={"Idempotency-Key": "batch-failure"}
    ).json()["generation"]
    original = EndpointEmbedding._request
    batches = 0

    def interrupted(
        self: EndpointEmbedding, texts: list[str], *, query: bool
    ) -> list[list[float]]:
        nonlocal batches
        if not query:
            batches += 1
            if batches == 2:
                raise EmbeddingUnavailableError
        return original(self, texts, query=query)

    with monkeypatch.context() as patch:
        patch.setattr(EndpointEmbedding, "_request", interrupted)
        with pytest.raises(ProcessingFailedError):
            complete(app, settings, {"generation_id": generation["id"]})
    assert batches == 2
    assert search(client, kb).json() == before
    with app.state.session_factory() as session:
        assert (
            session.get_one(ProcessingGeneration, UUID(generation["id"])).status
            == "failed"
        )
        assert (
            list(
                session.scalars(
                    select(Chunk).where(Chunk.generation_id == UUID(generation["id"]))
                )
            )
            == []
        )
        assert (
            session.get_one(DocumentVersion, UUID(uploaded["version"]["id"])).status
            == "ready"
        )

    def crash(*_args: object, **_kwargs: object) -> list[list[float]]:
        raise SystemExit("worker interrupted during embedding")

    with monkeypatch.context() as patch:
        patch.setattr(EndpointEmbedding, "_request", crash)
        with pytest.raises(SystemExit):
            complete(app, settings, {"generation_id": generation["id"]})
    assert search(client, kb).json() == before
    complete(app, settings, {"generation_id": generation["id"]})
    after = search(client, kb).json()
    assert {item["chunk_id"] for item in after["results"]}.isdisjoint(
        item["chunk_id"] for item in before["results"]
    )
    assert len(after["results"]) == len(before["results"])
    assert (
        client.get(base + f"/{uploaded['document']['id']}").json()[
            "active_processing_generation_id"
        ]
        == generation["id"]
    )


def test_provider_outage_is_http_503_and_liveness_survives(
    library: tuple[TestClient, Any, Settings, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app, settings, kb = library
    uploaded = client.post(
        f"/knowledge-bases/{kb}/documents", files={"file": ("a.txt", b"alpha")}
    ).json()
    complete(app, settings, uploaded)

    def failure(*_args: object, **_kwargs: object) -> list[list[float]]:
        raise EmbeddingUnavailableError("PRIVATE PROVIDER KEY")

    monkeypatch.setattr(EndpointEmbedding, "_request", failure)
    response = search(client, kb)
    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding provider is unavailable"}
    assert client.get("/health").status_code == 200


def test_legacy_ready_chunks_require_explicit_reprocessing(
    library: tuple[TestClient, Any, Settings, str],
) -> None:
    client, app, settings, kb = library
    base = f"/knowledge-bases/{kb}/documents"
    uploaded = client.post(base, files={"file": ("a.txt", b"alpha")}).json()
    complete(app, settings, uploaded)
    # Model the nullable index identity on retained #422 parsed/chunked history.
    with app.state.session_factory() as session:
        session.execute(delete(IndexedChunk))
        generation = session.get_one(
            ProcessingGeneration, UUID(uploaded["generation_id"])
        )
        generation.index_generation_id = None
        session.commit()
        assert len(list(session.scalars(select(Chunk)))) == 1
    path = base + f"/{uploaded['document']['id']}"
    assert client.get(path).json()["active_version_id"] is None
    assert search(client, kb).json()["results"] == []
    original_path = (
        settings.storage_path / f"originals/{kb}/{uploaded['version']['checksum']}"
    )
    original_path.unlink()
    generation = client.post(
        path + f"/versions/{uploaded['version']['id']}/reprocess",
        json={},
        headers={"Idempotency-Key": "index-legacy"},
    ).json()["generation"]
    complete(app, settings, {"generation_id": generation["id"]})
    assert search(client, kb).json()["results"][0]["text"] == "alpha"
    assert client.get(path).json()["active_version_id"] == uploaded["version"]["id"]


@pytest.mark.parametrize("provider", ["local", "openrouter"])
def test_provider_http_contract_document_and_query(
    library: tuple[TestClient, Any, Settings, str],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    client, app, settings, _kb = library
    settings.embedding_api_key = SecretStr("fixture-provider-token")
    kb = client.post(
        "/knowledge-bases",
        json={
            "name": "Provider",
            "embedding_provider": provider,
            "embedding_model": "Qwen3-Embedding-0.6B"
            if provider == "local"
            else "test/model",
            "embedding_dimensions": 2,
            "embedding_endpoint": "http://independent-host:8081/v1",
        },
    ).json()["id"]
    requests: list[dict[str, Any]] = []
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://independent-host:8081/v1/embeddings"
        assert request.headers["authorization"] == "Bearer fixture-provider-token"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]}
                    for i in reversed(range(len(body["input"])))
                ],
            },
        )

    monkeypatch.setattr(EndpointEmbedding, "_request", REAL_REQUEST)
    monkeypatch.setattr(
        "rag_service.services.embeddings.httpx.Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    uploaded = client.post(
        f"/knowledge-bases/{kb}/documents", files={"file": ("a.txt", b"alpha")}
    ).json()
    assert requests == []  # Upload is still asynchronous.
    complete(app, settings, uploaded)
    assert search(client, kb).status_code == 200
    assert requests[0]["input"] == ["alpha"]
    assert requests[0]["dimensions"] == 2
    if provider == "local":
        assert requests[1]["input"][0].startswith("Instruct:")
        assert requests[1]["input"][0].endswith("Query: alpha")
    else:
        assert requests[0]["input_type"] == "search_document"
        assert requests[1]["input_type"] == "search_query"
    assert "fixture-provider-token" not in repr(requests)
