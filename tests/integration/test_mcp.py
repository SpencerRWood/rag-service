"""Protocol-level MCP clients exercise the HTTP application's indexed library."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import SecretStr
from sqlalchemy.exc import SQLAlchemyError

from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.mcp import create_mcp
from rag_service.services.embeddings import EmbeddingUnavailableError, EndpointEmbedding
from rag_service.services.processing import process_generation
from rag_service.storage import build_storage


async def assert_read_only_contract(mcp: ClientSession) -> None:
    """Inspect the complete protocol tool surface, including annotations."""
    initialized = await mcp.initialize()
    assert initialized.capabilities.tools is not None
    tools = (await mcp.list_tools()).tools
    assert {tool.name for tool in tools} == {"search", "fetch"}
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.outputSchema is not None
    assert (await mcp.list_resources()).resources == []
    assert (await mcp.list_prompts()).prompts == []


async def assert_provider_outage(
    mcp: ClientSession,
    kb: str,
    arguments: dict[str, str],
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search fails safely while fetching retained evidence needs no provider."""

    def unavailable(*_args: object, **_kwargs: object) -> list[list[float]]:
        raise EmbeddingUnavailableError("private-provider-token")

    monkeypatch.setattr(EndpointEmbedding, "_request", unavailable)
    outage = await mcp.call_tool(
        "search", {"knowledge_base_id": kb, "request": {"query": "alpha"}}
    )
    assert outage.isError
    assert "Embedding provider is unavailable" in str(outage.content)
    assert "private-provider-token" not in str(outage)
    assert (await mcp.call_tool("fetch", arguments)).structuredContent == evidence


def test_mcp_http_search_fetch_and_read_only_contract(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database),
        storage_path=tmp_path,
        mcp_path="/retrieval/mcp",
        mcp_allowed_hosts=["testserver"],
    )
    app = create_app(settings)

    async def exercise() -> None:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as http,
        ):
            kb = (await http.post("/knowledge-bases", json={"name": "Library"})).json()[
                "id"
            ]
            base = f"/knowledge-bases/{kb}/documents"
            uploaded = (
                await http.post(
                    base,
                    files={"file": ("guide.md", b"# Alpha\nSource backed evidence")},
                    data={
                        "metadata": json.dumps(
                            {
                                "title": "Guide",
                                "tags": ["reference"],
                                "source": "manual",
                            }
                        )
                    },
                )
            ).json()
            with app.state.session_factory() as session:
                process_generation(
                    session,
                    build_storage(settings),
                    UUID(uploaded["generation_id"]),
                    settings,
                )
            other = (
                await http.post("/knowledge-bases", json={"name": "Other"})
            ).json()["id"]
            async with (
                streamable_http_client(
                    "http://testserver/retrieval/mcp", http_client=http
                ) as (read, write, _),
                ClientSession(read, write) as mcp,
            ):
                await assert_read_only_contract(mcp)
                request: dict[str, Any] = {
                    "query": "alpha",
                    "result_count": 1,
                    "filters": [
                        {"field": "tag", "operator": "in", "value": ["reference"]}
                    ],
                }
                expected = (
                    await http.post(f"/knowledge-bases/{kb}/retrieve", json=request)
                ).json()
                result = await mcp.call_tool(
                    "search", {"knowledge_base_id": kb, "request": request}
                )
                assert not result.isError
                assert result.structuredContent == expected
                chunk = expected["results"][0]
                arguments = {"knowledge_base_id": kb, "chunk_id": chunk["chunk_id"]}
                fetched = await mcp.call_tool("fetch", arguments)
                evidence = {
                    key: value for key, value in chunk.items() if key != "score"
                }
                assert fetched.structuredContent == evidence
                assert (
                    await http.get(f"/knowledge-bases/{kb}/chunks/{chunk['chunk_id']}")
                ).json() == evidence
                filtered = await mcp.call_tool(
                    "search",
                    {
                        "knowledge_base_id": kb,
                        "request": {
                            "query": "alpha",
                            "filters": [
                                {
                                    "field": "source",
                                    "operator": "eq",
                                    "value": "absent",
                                }
                            ],
                        },
                    },
                )
                assert filtered.structuredContent == {
                    "knowledge_base_id": kb,
                    "results": [],
                }
                for invalid in [
                    {"query": " "},
                    {"query": "alpha", "result_count": 101},
                    {
                        "query": "alpha",
                        "filters": [
                            {"field": "unknown", "operator": "eq", "value": "x"}
                        ],
                    },
                ]:
                    assert (
                        await mcp.call_tool(
                            "search", {"knowledge_base_id": kb, "request": invalid}
                        )
                    ).isError
                assert (await mcp.call_tool("upload", {})).isError
                assert (
                    await mcp.call_tool(
                        "fetch", {**arguments, "knowledge_base_id": other}
                    )
                ).isError
                assert (
                    await mcp.call_tool(
                        "fetch", {**arguments, "chunk_id": str(uuid4())}
                    )
                ).isError
                assert (
                    await mcp.call_tool(
                        "search",
                        {
                            "knowledge_base_id": other,
                            "request": {
                                "query": "alpha",
                                "version_id": chunk["version_id"],
                            },
                        },
                    )
                ).isError
                # A retained citation remains readable after reprocessing.
                path = base + f"/{chunk['document_id']}/versions/{chunk['version_id']}"
                generation = (
                    await http.post(
                        path + "/reprocess",
                        json={},
                        headers={"Idempotency-Key": "mcp-reprocess"},
                    )
                ).json()["generation"]
                with app.state.session_factory() as session:
                    process_generation(
                        session,
                        build_storage(settings),
                        UUID(generation["id"]),
                        settings,
                    )
                assert (
                    await mcp.call_tool("fetch", arguments)
                ).structuredContent == evidence

                await assert_provider_outage(mcp, kb, arguments, evidence, monkeypatch)
                await http.delete(base + f"/{chunk['document_id']}")
                assert (await mcp.call_tool("fetch", arguments)).isError
                assert (
                    await http.get(f"/knowledge-bases/{kb}/chunks/{chunk['chunk_id']}")
                ).status_code == 404
            assert (await http.get("/health")).status_code == 200
            blocked = await http.post(
                "/retrieval/mcp", headers={"Host": "untrusted.example"}, json={}
            )
            assert blocked.status_code == 421
            blocked_origin = await http.post(
                "/retrieval/mcp",
                headers={"Origin": "https://untrusted.example"},
                json={},
            )
            assert blocked_origin.status_code == 403

    anyio.run(exercise)


@pytest.mark.parametrize("configured", [False, True])
def test_mcp_safe_persistence_errors(
    configured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(
        Settings(database_url=SecretStr("sqlite://") if configured else SecretStr(""))
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise SQLAlchemyError("private-database-url")

    monkeypatch.setattr("rag_service.mcp.retrieve", fail)

    async def exercise() -> None:
        async with create_connected_server_and_client_session(app.state.mcp) as client:
            result = await client.call_tool(
                "search",
                {"knowledge_base_id": str(uuid4()), "request": {"query": "alpha"}},
            )
            assert result.isError
            expected = (
                "Persistence is unavailable"
                if configured
                else "Persistence is not configured"
            )
            assert expected in str(result.content)
            assert "private-database-url" not in str(result)

    anyio.run(exercise)


def test_mcp_without_database_needs_no_model_credentials() -> None:
    server = create_mcp(
        Settings(database_url=SecretStr(""), embedding_api_key=SecretStr("")), None
    )

    async def exercise() -> None:
        async with create_connected_server_and_client_session(server) as client:
            assert len((await client.list_tools()).tools) == 2

    anyio.run(exercise)
