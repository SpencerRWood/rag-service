"""OpenAI-compatible text embeddings with independent health and readiness."""

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_embedding.config import MODEL, Settings
from rag_embedding.model import Encoder, QwenEncoder

LOG = logging.getLogger("rag_embedding")
Text = Annotated[str, Field(strict=True, min_length=1, max_length=32768)]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["Qwen/Qwen3-Embedding-0.6B"] = MODEL
    input: Text | Annotated[list[Text], Field(min_length=1, max_length=256)]
    dimensions: Literal[1024] = 1024
    encoding_format: Literal["float"] = "float"
    user: str | None = Field(default=None, max_length=256)

    @field_validator("input")
    @classmethod
    def validate_texts(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value
        if any(not text.strip() for text in texts) or sum(map(len, texts)) > 262144:
            raise ValueError("Input must contain non-empty bounded text")
        return value


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {"message": message, "type": "embedding_error", "code": code}
        },
    )


def event(name: str, **fields: object) -> None:
    LOG.warning(json.dumps({"event": name, **fields}))


class RequestLimit:
    """Bound request body memory and upload time before JSON parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        try:
            async with asyncio.timeout(10):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > 1048576:
                        await error(413, "request_too_large", "Request exceeds 1 MiB")(
                            scope, receive, send
                        )
                        return
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await error(408, "request_timeout", "Request upload timed out")(
                scope, receive, send
            )
            return
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Settings], Encoder] = QwenEncoder,
) -> FastAPI:
    config = settings or Settings()
    encoder: Encoder | None = None
    state = "loading"
    busy = False
    pending: set[asyncio.Task[tuple[list[list[float]], int]]] = set()

    async def load() -> None:
        nonlocal encoder, state
        try:
            encoder = await asyncio.wait_for(
                asyncio.to_thread(loader, config), config.embedding_startup_timeout
            )
        except Exception as exc:  # noqa: BLE001 -- do not leak provider diagnostics
            state = "failed"
            event(
                "model_load_failed",
                model=config.embedding_model,
                exception_type=type(exc).__name__,
            )
        else:
            state = "ready"
            event("model_loaded", model=config.embedding_model, dimensions=1024)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(load())
        yield
        await task
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(title="RAG Embedding", lifespan=lifespan)
    app.add_middleware(RequestLimit)

    @app.exception_handler(RequestValidationError)
    async def invalid(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return error(422, "invalid_request", "Invalid embedding request")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return JSONResponse(
            status_code=200 if state == "ready" else 503,
            content={
                "status": state,
                "model": config.embedding_model,
                "model_revision": config.embedding_model_revision,
                "dimensions": config.embedding_dimensions,
                "source_revision": config.source_revision,
                "release_revision": config.release_revision,
            },
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        if state != "ready":
            return error(503, "model_not_ready", "Embedding model is unavailable")
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": config.embedding_model,
                        "object": "model",
                        "owned_by": "Qwen",
                    }
                ],
            }
        )

    async def infer(texts: list[str], model: Encoder) -> tuple[list[list[float]], int]:
        nonlocal busy
        try:
            return await asyncio.to_thread(model.encode, texts)
        finally:
            busy = False

    def finished(task: asyncio.Task[tuple[list[list[float]], int]]) -> None:
        pending.discard(task)
        if not task.cancelled():
            task.exception()

    @app.post("/v1/embeddings")
    async def embeddings(payload: EmbeddingRequest) -> JSONResponse:
        nonlocal busy
        if state != "ready" or encoder is None:
            return error(503, "model_not_ready", "Embedding model is unavailable")
        if busy:
            return error(429, "busy", "Embedding service is busy; retry later")
        texts = [payload.input] if isinstance(payload.input, str) else payload.input
        busy = True
        task = asyncio.create_task(infer(texts, encoder))
        pending.add(task)
        task.add_done_callback(finished)
        started = monotonic()
        try:
            vectors, tokens = await asyncio.wait_for(
                asyncio.shield(task), timeout=config.embedding_timeout
            )
            if len(vectors) != len(texts) or any(
                len(vector) != 1024
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
                for vector in vectors
            ):
                raise ValueError("Invalid model output")
        except TimeoutError:
            event("inference_timeout", batch_size=len(texts))
            return error(504, "inference_timeout", "Embedding request timed out")
        except Exception:  # noqa: BLE001 -- safe model boundary
            event("inference_failed", batch_size=len(texts))
            return error(500, "inference_failed", "Embedding inference failed")
        event(
            "inference_complete", batch_size=len(texts), seconds=monotonic() - started
        )
        return JSONResponse(
            content={
                "object": "list",
                "model": config.embedding_model,
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    return app
