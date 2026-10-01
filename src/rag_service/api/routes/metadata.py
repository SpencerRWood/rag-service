"""Public build identity without exposing application settings."""

from typing import cast

from fastapi import APIRouter, Request

from rag_service.config import Settings

router = APIRouter()


@router.get("/version", tags=["health"])
def read_version(request: Request) -> dict[str, str]:
    """Expose package version and infrastructure-injected source identity."""
    settings = cast(Settings, request.app.state.settings)
    return {
        "service": "rag-service",
        "version": request.app.version,
        "source_revision": settings.source_revision,
        "release_revision": settings.release_revision,
    }
