"""Exercise the shipped verifier with real API/MCP and simulated Dagster state."""

import json
from pathlib import Path
from uuid import UUID

import anyio
import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from rag_service import diagnostics, verification
from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.models.persistence import Document, ProcessingGeneration
from rag_service.services.processing import process_generation
from rag_service.storage import build_storage


@pytest.mark.parametrize("boundary_allowed", [False, True])
def test_dev_verifier_checks_protocol_and_trusted_boundary(
    boundary_allowed: bool,
    migrated_database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database),
        storage_path=tmp_path,
        source_revision="source",
        release_revision="release",
        mcp_allowed_hosts=["testserver"],
    )
    app = create_app(settings)
    monkeypatch.setattr(diagnostics, "dagster_health", lambda _settings: None)
    monkeypatch.setattr(diagnostics, "embedding_health", lambda _settings: None)
    for name, value in {
        "RAG_VERIFY_URL": "http://testserver",
        "RAG_VERIFY_DAGSTER_URL": "http://dagster",
        "RAG_VERIFY_UNTRUSTED_MCP_URL": "http://edge/mcp",
        "RAG_VERIFY_VERSION": app.version,
        "RAG_VERIFY_SOURCE_REVISION": "source",
        "RAG_VERIFY_RELEASE_REVISION": "release",
    }.items():
        monkeypatch.setenv(name, value)
    original = httpx.AsyncClient
    transport = httpx.ASGITransport(app=app)
    run_checks = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal run_checks
        if request.url.host == "edge":
            return httpx.Response(200 if boundary_allowed else 403)
        if request.url.host == "dagster":
            run_id = json.loads(request.content)["variables"]["id"]
            run_checks += 1
            if run_checks == 1:
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "runOrError": {
                                "__typename": "Run",
                                "runId": run_id,
                                "status": "STARTED",
                            }
                        }
                    },
                )
            with app.state.session_factory() as session:
                generation = session.scalar(select(ProcessingGeneration.id))
                assert isinstance(generation, UUID)
                process_generation(
                    session, build_storage(settings), generation, settings
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "runOrError": {
                            "__typename": "Run",
                            "runId": run_id,
                            "status": "SUCCESS",
                        }
                    }
                },
            )
        return await transport.handle_async_request(request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            if boundary_allowed:
                with pytest.raises(ValueError, match="assertion failed"):
                    await verification.verify()
            else:
                result = await verification.verify()
                assert result["status"] == "passed"
                checks = result["checks"]
                assert isinstance(checks, list)
                assert "mcp_search_fetch" in checks
                with app.state.session_factory() as session:
                    document = session.scalar(select(Document))
                    assert document is not None
                    assert document.deleted_at is not None

    anyio.run(exercise)


def test_missing_verification_context_is_safe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RAG_VERIFY_URL", raising=False)
    assert verification.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "unavailable",
        "missing_setting": "RAG_VERIFY_URL",
    }


def test_verification_failure_excludes_exception_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def failed() -> dict[str, object]:
        raise ValueError("private-provider-token")

    monkeypatch.setattr(verification, "verify", failed)
    assert verification.main() == 1
    output = capsys.readouterr().out
    assert '"status": "failed"' in output
    assert "private-provider-token" not in output


def test_verification_cli_returns_bounded_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def verified() -> dict[str, object]:
        return {"status": "passed"}

    monkeypatch.setattr(verification, "verify", verified)
    assert verification.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "passed"}
