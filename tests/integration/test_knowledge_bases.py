"""HTTP behavior, persistence across restarts, and tenant isolation."""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.models.persistence import DEFAULT_TENANT_ID, KnowledgeBase, Tenant


@pytest.fixture
def client(migrated_database: str) -> Iterator[TestClient]:
    settings = Settings(database_url=SecretStr(migrated_database))
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.session_factory.kw["bind"].dispose()


def test_create_list_get_and_restart(
    client: TestClient, migrated_database: str
) -> None:
    response = client.post("/knowledge-bases", json={"name": "Engineering"})
    assert response.status_code == 201
    record = response.json()
    assert record["embedding_provider"] == "local"
    assert record["embedding_model"] == "Qwen3-Embedding-0.6B"
    assert record["embedding_dimensions"] == 1024
    assert record["chunk_size"] == 512
    assert record["chunk_overlap"] == 64
    assert "embedding_api_key" not in record
    assert client.get("/knowledge-bases").json() == [record]
    assert client.get(f"/knowledge-bases/{record['id']}").json() == record
    restarted = create_app(Settings(database_url=SecretStr(migrated_database)))
    with TestClient(restarted) as new_client:
        assert new_client.get(f"/knowledge-bases/{record['id']}").json() == record
    restarted.state.session_factory.kw["bind"].dispose()


def test_effective_overrides_and_validation(client: TestClient) -> None:
    response = client.post(
        "/knowledge-bases",
        json={
            "name": " Overrides ",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "embedding_provider": "openrouter",
            "embedding_model": "provider/model",
            "embedding_dimensions": 768,
            "embedding_endpoint": "https://example.com/embeddings",
        },
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Overrides"
    assert response.json()["embedding_dimensions"] == 768
    assert (
        client.post("/knowledge-bases", json={"name": "Overrides"}).status_code == 409
    )
    for invalid in [
        {"name": " "},
        {"name": "bad", "chunk_size": 32},
        {"name": "bad", "embedding_dimensions": 0},
        {"name": "bad", "embedding_provider": "unknown"},
        {"name": "bad", "extra": True},
    ]:
        assert client.post("/knowledge-bases", json=invalid).status_code == 422
    assert client.get(f"/knowledge-bases/{uuid4()}").status_code == 404
    assert client.get("/knowledge-bases/not-a-uuid").status_code == 422
    assert client.get("/knowledge-bases?limit=101").status_code == 422
    assert client.get("/knowledge-bases?offset=1").json() == []


def test_default_tenant_and_isolation(
    client: TestClient, migrated_database: str
) -> None:
    engine = create_engine(migrated_database)
    try:
        with Session(engine) as session:
            assert session.get(Tenant, DEFAULT_TENANT_ID) is not None
            other = Tenant(name="Other tenant")
            session.add(other)
            session.flush()
            settings = Settings()
            hidden = KnowledgeBase(
                tenant_id=other.id,
                name="private",
                chunk_size=512,
                chunk_overlap=64,
                embedding_provider="local",
                embedding_model=settings.embedding_model,
                embedding_dimensions=1024,
                embedding_endpoint=settings.embedding_endpoint,
            )
            session.add(hidden)
            session.commit()
            hidden_id = hidden.id
            assert session.scalar(select(KnowledgeBase.id)) == hidden_id
        assert client.get("/knowledge-bases").json() == []
        assert client.get(f"/knowledge-bases/{hidden_id}").status_code == 404
    finally:
        engine.dispose()


def test_liveness_and_metadata_do_not_require_database() -> None:
    with TestClient(
        create_app(
            Settings(
                database_url=SecretStr(""),
                source_revision="abc123",
                release_revision="v0.0.1",
            )
        )
    ) as test_client:
        assert test_client.get("/health").status_code == 200
        metadata = test_client.get("/version").json()
        assert metadata["source_revision"] == "abc123"
        assert metadata["release_revision"] == "v0.0.1"
        assert test_client.get("/knowledge-bases").status_code == 503


def test_unmigrated_database_returns_safe_error(tmp_path: Path) -> None:
    app = create_app(
        Settings(database_url=SecretStr(f"sqlite:///{tmp_path / 'empty.db'}"))
    )
    with TestClient(app) as test_client:
        response = test_client.get("/knowledge-bases")
        assert response.status_code == 503
        assert response.json() == {"detail": "Persistence is unavailable"}
        assert test_client.get("/health").status_code == 200
    app.state.session_factory.kw["bind"].dispose()
