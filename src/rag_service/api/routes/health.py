"""Liveness endpoint for the RAG retrieval API."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def read_health() -> dict[str, str]:
    """Return a lightweight liveness response without touching dependencies."""
    return {"service": "rag-service", "status": "ok"}
