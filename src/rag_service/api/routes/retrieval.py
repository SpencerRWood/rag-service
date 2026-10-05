"""HTTP retrieval transport; consumers never need direct vector-schema access."""

from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from rag_service.api.routes.knowledge_bases import SessionDependency
from rag_service.config import Settings
from rag_service.models.retrieval import RetrievalRequest, RetrievalResponse
from rag_service.services.documents import DocumentNotFoundError
from rag_service.services.embeddings import EmbeddingUnavailableError
from rag_service.services.retrieval import retrieve

router = APIRouter(prefix="/knowledge-bases", tags=["retrieval"])


@router.post("/{knowledge_base_id}/retrieve", response_model=RetrievalResponse)
def retrieve_context(
    knowledge_base_id: UUID,
    payload: RetrievalRequest,
    request: Request,
    session: SessionDependency,
) -> RetrievalResponse:
    """Return ranked evidence without generative inference."""
    try:
        return retrieve(
            session,
            knowledge_base_id,
            payload,
            cast(Settings, request.app.state.settings),
        )
    except DocumentNotFoundError:
        raise HTTPException(404, "Knowledge base or source version not found") from None
    except EmbeddingUnavailableError:
        raise HTTPException(503, "Embedding provider is unavailable") from None
