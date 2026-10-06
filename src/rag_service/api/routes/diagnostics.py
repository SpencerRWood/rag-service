"""Private-ingress operational endpoints; content diagnostics use retrieval."""

from typing import cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from rag_service.config import Settings
from rag_service.diagnostics import dependencies

router = APIRouter(tags=["diagnostics"])


@router.get("/ready")
def read_readiness(request: Request) -> JSONResponse:
    """A provider outage degrades ingestion/search without failing core readiness."""
    states = dependencies(
        cast(Settings, request.app.state.settings), request.app.state.session_factory
    )
    ready = all(states[name] == "available" for name in ("database", "storage"))
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "dependencies": states},
        status_code=200 if ready else 503,
    )


@router.get("/metrics")
def read_metrics(request: Request) -> Response:
    """A failed durable collector is an unavailable scrape, never zero success."""
    try:
        registry = cast(CollectorRegistry, request.app.state.metrics_registry)
        if request.app.state.session_factory is None:
            raise ValueError("Persistence is not configured")
        content = generate_latest(registry)
    except Exception:
        raise HTTPException(503, "Metrics are unavailable") from None
    return Response(content, headers={"Content-Type": CONTENT_TYPE_LATEST})
