"""Dependency outages, cross-process measurements, and safe diagnostic contracts."""

from pathlib import Path
from typing import Literal
from uuid import UUID

import boto3
import httpx
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr
from sqlalchemy import select

from rag_service import diagnostics
from rag_service.config import Settings
from rag_service.dagster.sensors import mark_interrupted
from rag_service.main import create_app
from rag_service.models.persistence import ProcessingAttempt, ProcessingObservation
from rag_service.services.embeddings import EmbeddingUnavailableError, EndpointEmbedding
from rag_service.services.processing import ProcessingFailedError, process_generation
from rag_service.storage import build_storage


def fail(*_args: object, **_kwargs: object) -> None:
    raise EmbeddingUnavailableError("private-database-password and document-content")


def test_readiness_core_and_optional_dependencies(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    monkeypatch.setattr(diagnostics, "dagster_health", fail)
    monkeypatch.setattr(diagnostics, "embedding_health", fail)
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "dependencies": {
                "database": "available",
                "storage": "available",
                "dagster": "unavailable",
                "embedding": "unavailable",
            },
        }
        assert list(tmp_path.glob("tmp*")) == []
        monkeypatch.setattr(diagnostics, "database_health", fail)
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["dependencies"]["database"] == "unavailable"
        assert "private" not in response.text
        assert client.get("/health").status_code == 200


def test_missing_configuration_and_failed_metric_scrape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics, "dagster_health", fail)
    monkeypatch.setattr(diagnostics, "embedding_health", fail)
    with TestClient(
        create_app(
            Settings(database_url=SecretStr(""), storage_path=tmp_path / "absent")
        )
    ) as client:
        states = client.get("/ready").json()["dependencies"]
        assert states["database"] == "not_configured"
        assert states["storage"] == "unavailable"
        assert client.get("/metrics").status_code == 503
    with TestClient(
        create_app(Settings(database_url=SecretStr("sqlite://")))
    ) as client:
        assert client.get("/metrics").json() == {"detail": "Metrics are unavailable"}


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("provider", ["local", "openrouter"])
def test_dependency_catalogs(
    available: bool,
    provider: Literal["local", "openrouter"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        embedding_provider=provider, embedding_api_key=SecretStr("private-token")
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            assert request.url.path == (
                "/v1/embeddings/models" if provider == "openrouter" else "/v1/models"
            )
            assert request.headers["Authorization"] == "Bearer private-token"
            return httpx.Response(
                200,
                json={
                    "data": [{"id": settings.embedding_model if available else "other"}]
                },
            )
        assert request.url.path.endswith("/graphql")
        return httpx.Response(
            200,
            json={
                "data": {
                    "repositoriesOrError": {
                        "__typename": "RepositoryConnection",
                        "nodes": [
                            {
                                "location": {
                                    "name": settings.dagster_location
                                    if available
                                    else "other"
                                }
                            }
                        ],
                    }
                }
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    expected = "available" if available else "unavailable"
    assert diagnostics.probe(lambda: diagnostics.embedding_health(settings)) == expected
    assert diagnostics.probe(lambda: diagnostics.dagster_health(settings)) == expected
    assert (
        diagnostics.probe(
            lambda: diagnostics.embedding_health(
                Settings(embedding_provider="openrouter")
            )
        )
        == "unavailable"
    )


@mock_aws
def test_s3_readiness_uses_configured_bucket() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="rag-diagnostics")
    settings = Settings(storage_backend="s3", s3_bucket="rag-diagnostics")
    assert (
        diagnostics.probe(lambda: diagnostics.storage_health(settings)) == "available"
    )
    assert (
        diagnostics.probe(
            lambda: diagnostics.storage_health(
                Settings(storage_backend="s3", s3_bucket="missing")
            )
        )
        == "unavailable"
    )
    assert client.list_objects_v2(Bucket="rag-diagnostics")["KeyCount"] == 0


def metric_value(
    content: str, name: str, labels: dict[str, str] | None = None
) -> float:
    return sum(
        sample.value
        for family in text_string_to_metric_families(content)
        for sample in family.samples
        if sample.name == name and (labels is None or sample.labels == labels)
    )


def test_worker_metrics_survive_api_restart_and_count_failed_embeddings(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)
    with TestClient(app) as client:
        kb = client.post("/knowledge-bases", json={"name": "metrics"}).json()["id"]
        private_text = b"private-source-content"
        base = f"/knowledge-bases/{kb}/documents"
        uploaded = client.post(
            base, files={"file": ("private-name.txt", private_text)}
        ).json()
        generation = UUID(uploaded["generation_id"])
        with app.state.session_factory() as session:
            process_generation(session, build_storage(settings), generation, settings)
            process_generation(session, build_storage(settings), generation, settings)
            observations = list(session.scalars(select(ProcessingObservation)))
            assert len(observations) == 1
            assert observations[0].chunk_count > 0
        assert (
            client.post(
                f"/knowledge-bases/{kb}/retrieve", json={"query": "private-query"}
            ).status_code
            == 200
        )
        monkeypatch.setattr(EndpointEmbedding, "_request", fail)
        assert (
            client.post(
                f"/knowledge-bases/{kb}/retrieve", json={"query": "private-query"}
            ).status_code
            == 503
        )
        other = client.post(
            base, files={"file": ("failed.txt", b"different-private-content")}
        ).json()
        with (
            app.state.session_factory() as session,
            pytest.raises(ProcessingFailedError),
        ):
            process_generation(
                session, build_storage(settings), UUID(other["generation_id"]), settings
            )
    # New API factory sees measurements committed by independent worker sessions.
    with TestClient(create_app(settings)) as restarted:
        response = restarted.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert (
            metric_value(content, "rag_processing_runs_total", {"outcome": "success"})
            == 1
        )
        assert (
            metric_value(content, "rag_processing_runs_total", {"outcome": "failure"})
            == 1
        )
        assert (
            metric_value(
                content,
                "rag_document_embedding_duration_seconds_count",
                {"outcome": "failure"},
            )
            == 1
        )
        assert metric_value(content, "rag_processing_duration_seconds_sum") > 0
        assert (
            metric_value(
                content, "rag_retrieval_requests_total", {"outcome": "failure"}
            )
            > 0
        )
        assert metric_value(content, "rag_retrieval_similarity_score_count") > 0
        assert (
            metric_value(content, "rag_upload_requests_total", {"outcome": "success"})
            >= 2
        )
        assert "private-" not in content
        assert kb not in content
        assert "text/plain" in response.headers["content-type"]


def test_interrupted_worker_has_unknown_duration_and_counts_once(
    migrated_database: str, tmp_path: Path
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)
    with TestClient(app) as client:
        kb = client.post("/knowledge-bases", json={"name": "interrupted"}).json()["id"]
        uploaded = client.post(
            f"/knowledge-bases/{kb}/documents",
            files={"file": ("source.txt", b"fixture")},
        ).json()
        generation_id = UUID(uploaded["generation_id"])
        with app.state.session_factory() as session:
            attempts = set(session.scalars(select(ProcessingAttempt.id)))
            mark_interrupted(session, generation_id, attempts)
            mark_interrupted(session, generation_id, attempts)
            observations = list(session.scalars(select(ProcessingObservation)))
            assert len(observations) == 1
            assert observations[0].duration_seconds is None
        content = client.get("/metrics").text
        assert (
            metric_value(content, "rag_processing_runs_total", {"outcome": "failure"})
            == 1
        )
        assert (
            metric_value(
                content, "rag_processing_duration_seconds_count", {"outcome": "failure"}
            )
            == 0
        )
