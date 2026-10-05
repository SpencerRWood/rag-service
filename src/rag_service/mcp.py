"""Read-only MCP adapter over the same evidence services used by HTTP."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from rag_service.config import Settings
from rag_service.models.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    SourceEvidence,
)
from rag_service.services.documents import DocumentNotFoundError
from rag_service.services.embeddings import EmbeddingUnavailableError
from rag_service.services.retrieval import fetch_chunk, retrieve


def create_mcp(
    settings: Settings, session_factory: sessionmaker[Session] | None
) -> FastMCP:
    """Build stateless Streamable HTTP tools with explicit read-only annotations."""
    server = FastMCP(
        "RAG Service",
        instructions="Retrieve source evidence for cited answers.",
        stateless_http=True,
        json_response=True,
        streamable_http_path=settings.mcp_path,
        transport_security=TransportSecuritySettings(
            allowed_hosts=settings.mcp_allowed_hosts,
            allowed_origins=settings.mcp_allowed_origins,
        ),
    )

    @contextmanager
    def session_scope() -> Iterator[Session]:
        if session_factory is None:
            raise ToolError("Persistence is not configured")
        try:
            with session_factory() as session:
                yield session
        except DocumentNotFoundError:
            raise ToolError("Knowledge base or indexed source not found") from None
        except EmbeddingUnavailableError:
            raise ToolError("Embedding provider is unavailable") from None
        except SQLAlchemyError:
            raise ToolError("Persistence is unavailable") from None

    annotations = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(annotations=annotations)
    async def search(
        knowledge_base_id: UUID, request: RetrievalRequest
    ) -> RetrievalResponse:
        """Search indexed evidence using query, count, metadata filters and history."""

        def execute() -> RetrievalResponse:
            with session_scope() as session:
                return retrieve(session, knowledge_base_id, request, settings)

        return await to_thread.run_sync(execute)

    @server.tool(annotations=annotations)
    async def fetch(knowledge_base_id: UUID, chunk_id: UUID) -> SourceEvidence:
        """Read one search result's exact chunk and provenance, without embedding."""

        def execute() -> SourceEvidence:
            with session_scope() as session:
                return fetch_chunk(session, knowledge_base_id, chunk_id)

        return await to_thread.run_sync(execute)

    return server
