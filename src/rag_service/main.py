"""Application factory for the RAG retrieval API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from sqlalchemy import Engine

from rag_service.api.router import api_router
from rag_service.config import Settings, load_settings
from rag_service.database import create_session_factory
from rag_service.mcp import create_mcp
from rag_service.telemetry import metrics_registry


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""
    settings = settings or load_settings()
    session_factory = (
        create_session_factory(settings)
        if settings.database_url.get_secret_value()
        else None
    )
    mcp = create_mcp(settings, session_factory)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            engine = session_factory.kw.get("bind") if session_factory else None
            if isinstance(engine, Engine):
                engine.dispose()

    app = FastAPI(
        title="RAG Service", version=version("rag-service"), lifespan=lifespan
    )
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.mcp = mcp
    app.state.metrics_registry = metrics_registry(session_factory)
    app.include_router(api_router)
    app.mount("/", mcp_app)
    return app


app = create_app()
