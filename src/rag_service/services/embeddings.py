"""LlamaIndex embedding contract for independently hosted embedding-only APIs."""

import asyncio
import math
from typing import Annotated, Literal, cast

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from rag_service.config import Settings
from rag_service.models.persistence import IndexGeneration


class EmbeddingUnavailableError(Exception):
    """Safe boundary for transport, authentication, and malformed model responses."""


class EmbeddingItem(BaseModel):
    """Strict vector values prevent coercion of provider corruption."""

    index: int = Field(ge=0, strict=True)
    embedding: list[Annotated[float, Field(strict=True)]] = Field(strict=True)
    model_config = ConfigDict(allow_inf_nan=False)


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response; usage extensions are ignored."""

    data: list[EmbeddingItem]
    model: str


class EndpointEmbedding(BaseEmbedding):
    """One document/query interface, without loading a model in either RAG role."""

    provider: Literal["local", "openrouter"]
    dimensions: int
    endpoint: str = Field(exclude=True, repr=False)
    timeout: float = 30
    _api_key: SecretStr = PrivateAttr(default_factory=lambda: SecretStr(""))

    def _request(self, texts: list[str], *, query: bool) -> list[list[float]]:
        inputs = texts
        if self.provider == "local" and query:
            inputs = [
                "Instruct: Given a web search query, retrieve relevant passages "
                f"that answer the query\nQuery: {text}"
                for text in texts
            ]
        headers = {}
        key = self._api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.provider == "openrouter" and not key:
            raise EmbeddingUnavailableError
        payload: dict[str, object] = {
            "model": self.model_name,
            "input": inputs,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        if self.provider == "openrouter":
            payload["input_type"] = "search_query" if query else "search_document"
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.post(
                    self.endpoint.rstrip("/") + "/embeddings",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = EmbeddingResponse.model_validate(response.json())
            ordered = sorted(result.data, key=lambda item: item.index)
            if result.model != self.model_name or [
                item.index for item in ordered
            ] != list(range(len(texts))):
                raise ValueError("Embedding identity mismatch")
            vectors = [item.embedding for item in ordered]
            if any(
                len(vector) != self.dimensions
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
                for vector in vectors
            ):
                raise ValueError("Invalid vector")
        except Exception:
            raise EmbeddingUnavailableError from None
        return vectors

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._request([query], query=True)[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._request([text], query=False)[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._request(texts, query=False)


def build_embedding(index: IndexGeneration, settings: Settings) -> EndpointEmbedding:
    """Use generation identity, never mutable defaults, for documents and queries."""
    provider = EndpointEmbedding(
        provider=cast(Literal["local", "openrouter"], index.provider),
        model_name=index.model,
        dimensions=index.dimensions,
        endpoint=index.endpoint,
        embed_batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout,
    )
    provider._api_key = settings.embedding_api_key
    return provider
