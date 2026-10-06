"""Verify private dev HTTP/Dagster/MCP behavior; output only bounded safe facts."""

import asyncio
import json
import os
import sys
from time import monotonic
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

CONTENT = b"RAG verification: immutable source evidence preserves document provenance."
RUN_QUERY = """
query($id: ID!) { runOrError(runId: $id) { __typename
  ... on Run { runId status }
} }
"""


class VerificationContextError(Exception):
    """Expose only the missing non-secret setting name."""


def required(name: str) -> str:
    """Missing runtime selectors fail clearly without configuration fallbacks."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise VerificationContextError(name)
    return value


def require(condition: bool) -> None:
    """Do not emit remote response bodies or credential-bearing URLs on failure."""
    if not condition:
        raise ValueError("Verification assertion failed")


async def verify() -> dict[str, object]:  # noqa: PLR0915 -- bounded end-to-end sequence
    """Unique fixture identities plus soft deletion make repeated runs safe."""
    base = required("RAG_VERIFY_URL").rstrip("/")
    dagster = required("RAG_VERIFY_DAGSTER_URL").rstrip("/")
    untrusted = required("RAG_VERIFY_UNTRUSTED_MCP_URL")
    expected = {
        "version": required("RAG_VERIFY_VERSION"),
        "source_revision": required("RAG_VERIFY_SOURCE_REVISION"),
        "release_revision": required("RAG_VERIFY_RELEASE_REVISION"),
    }
    document_path: str | None = None
    async with httpx.AsyncClient(timeout=3, follow_redirects=False) as http:
        version = await http.get(base + "/version")
        version.raise_for_status()
        require(
            all(version.json().get(key) == value for key, value in expected.items())
        )
        ready = await http.get(base + "/ready")
        ready.raise_for_status()
        require(
            all(state == "available" for state in ready.json()["dependencies"].values())
        )
        blocked = await http.post(
            untrusted, headers={"X-Forwarded-For": "192.168.1.21"}, json={}
        )
        require(blocked.status_code == 403)
        metrics = await http.get(base + "/metrics")
        metrics.raise_for_status()
        require("rag_retrieval_requests" in metrics.text)
        knowledge_base = await http.post(
            base + "/knowledge-bases", json={"name": "rag-verification-" + uuid4().hex}
        )
        knowledge_base.raise_for_status()
        kb = knowledge_base.json()["id"]
        try:
            uploaded = await http.post(
                base + f"/knowledge-bases/{kb}/documents",
                files={"file": ("verification.txt", CONTENT, "text/plain")},
            )
            uploaded.raise_for_status()
            source = uploaded.json()
            document_path = (
                base + f"/knowledge-bases/{kb}/documents/{source['document']['id']}"
            )
            run_id = source["dagster_run_id"]
            require(source["launch_submitted"] and bool(run_id))
            deadline = monotonic() + 35
            while True:
                response = await http.post(
                    dagster + "/graphql",
                    json={"query": RUN_QUERY, "variables": {"id": run_id}},
                )
                response.raise_for_status()
                run = response.json()["data"]["runOrError"]
                require(run["__typename"] == "Run" and run["runId"] == run_id)
                if run["status"] == "SUCCESS":
                    break
                require(
                    run["status"] not in {"FAILURE", "CANCELED"}
                    and monotonic() < deadline
                )
                await asyncio.sleep(1)
            query = {"query": "immutable source evidence", "result_count": 1}
            retrieved = await http.post(
                base + f"/knowledge-bases/{kb}/retrieve", json=query
            )
            retrieved.raise_for_status()
            result = retrieved.json()
            require(len(result["results"]) == 1)
            chunk = result["results"][0]
            require(chunk["text"] == CONTENT.decode())
            require(
                chunk["document_id"] == source["document"]["id"]
                and chunk["version_id"] == source["version"]["id"]
            )
            require(
                chunk["source"]["checksum"] == source["version"]["checksum"]
                and bool(chunk["source"]["location"])
            )
            async with (
                streamable_http_client(base + "/mcp", http_client=http) as (
                    read,
                    write,
                    _,
                ),
                ClientSession(read, write) as mcp,
            ):
                await mcp.initialize()
                tools = (await mcp.list_tools()).tools
                require({tool.name for tool in tools} == {"search", "fetch"})
                require(
                    all(
                        tool.annotations is not None
                        and tool.annotations.readOnlyHint is True
                        and tool.annotations.destructiveHint is False
                        for tool in tools
                    )
                )
                searched = await mcp.call_tool(
                    "search", {"knowledge_base_id": kb, "request": query}
                )
                require(not searched.isError and searched.structuredContent == result)
                fetched = await mcp.call_tool(
                    "fetch", {"knowledge_base_id": kb, "chunk_id": chunk["chunk_id"]}
                )
                require(
                    not fetched.isError
                    and fetched.structuredContent
                    == {key: value for key, value in chunk.items() if key != "score"}
                )
            return {
                "status": "passed",
                "knowledge_base_id": kb,
                "dagster_run_id": run_id,
                "source_revision": expected["source_revision"],
                "checks": [
                    "identity",
                    "readiness",
                    "private_boundary",
                    "metrics",
                    "api_launched_processing",
                    "http_retrieval",
                    "mcp_search_fetch",
                    "read_only_tools",
                ],
            }
        finally:
            if document_path:
                deleted = await http.delete(document_path)
                require(deleted.status_code == 204)


def main() -> int:
    """Safe CLI errors avoid tracing provider exceptions or source content."""
    try:
        result = asyncio.run(verify())
    except VerificationContextError as exc:
        sys.stdout.write(
            json.dumps({"status": "unavailable", "missing_setting": str(exc)}) + "\n"
        )
        return 1
    except Exception:
        sys.stdout.write(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "Dev verification failed; check private runtime context",
                }
            )
            + "\n"
        )
        return 1
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
