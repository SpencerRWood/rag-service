"""Provider corruption and transport failures remain safe and retryable."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from rag_service.services.embeddings import EmbeddingUnavailableError, EndpointEmbedding

REAL_REQUEST = EndpointEmbedding._request


@pytest.mark.parametrize(
    "response_body",
    [
        {},
        {"model": "wrong", "data": [{"index": 0, "embedding": [1, 0]}]},
        {"model": "model", "data": []},
        {"model": "model", "data": [{"index": 1, "embedding": [1, 0]}]},
        {"model": "model", "data": [{"index": 0, "embedding": [1]}]},
        {"model": "model", "data": [{"index": 0, "embedding": [0, 0]}]},
        {"model": "model", "data": [{"index": 0, "embedding": [float("nan"), 1]}]},
        {"model": "model", "data": [{"index": 0, "embedding": [float("inf"), 1]}]},
        {"model": "model", "data": [{"index": 0, "embedding": ["secret", 1]}]},
        {"model": "model", "data": [{"index": 0, "embedding": [True, 1]}]},
        {
            "model": "model",
            "data": [
                {"index": 0, "embedding": [1, 0]},
                {"index": 0, "embedding": [1, 0]},
            ],
        },
    ],
)
def test_invalid_provider_output_is_safe(
    monkeypatch: pytest.MonkeyPatch, response_body: dict[str, Any]
) -> None:
    provider = EndpointEmbedding(
        provider="local",
        dimensions=2,
        endpoint="http://embedding/v1",
        model_name="model",
    )
    monkeypatch.setattr(EndpointEmbedding, "_request", REAL_REQUEST)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            content=json.dumps(response_body),
            request=httpx.Request("POST", "http://embedding/v1/embeddings"),
        ),
    )
    with pytest.raises(EmbeddingUnavailableError) as error:
        provider.get_text_embedding("PRIVATE DOCUMENT")
    assert not str(error.value)
    assert error.value.__cause__ is None


def test_timeout_auth_and_async_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EndpointEmbedding(
        provider="openrouter",
        dimensions=2,
        endpoint="http://embedding/v1",
        model_name="model",
    )
    monkeypatch.setattr(EndpointEmbedding, "_request", REAL_REQUEST)
    with pytest.raises(EmbeddingUnavailableError):
        provider.get_query_embedding("PRIVATE QUERY")


def test_batch_order_and_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EndpointEmbedding(
        provider="local",
        dimensions=2,
        endpoint="http://embedding/v1",
        model_name="model",
    )
    monkeypatch.setattr(EndpointEmbedding, "_request", REAL_REQUEST)

    def respond(
        _self: object, _url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        del headers
        inputs = json["input"]
        return httpx.Response(
            200,
            json={
                "model": "model",
                "data": [
                    {"index": i, "embedding": [float(i + 1), 0.0]}
                    for i in reversed(range(len(inputs)))
                ],
            },
            request=httpx.Request("POST", "http://embedding/v1/embeddings"),
        )

    monkeypatch.setattr(httpx.Client, "post", respond)
    assert provider.get_text_embedding_batch(["a", "b"]) == [[1.0, 0.0], [2.0, 0.0]]
    assert asyncio.run(provider.aget_query_embedding("a")) == [1.0, 0.0]
    provider._api_key = SecretStr("PRIVATE KEY")

    def timeout(*_args: object, **_kwargs: object) -> httpx.Response:
        raise httpx.ReadTimeout("PRIVATE KEY / QUERY")

    monkeypatch.setattr(httpx.Client, "post", timeout)
    with pytest.raises(EmbeddingUnavailableError) as failure:
        asyncio.run(provider.aget_query_embedding("PRIVATE QUERY"))
    assert "PRIVATE" not in repr(provider)
    assert not str(failure.value)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *_args, **_kwargs: httpx.Response(
            401,
            text="PRIVATE KEY",
            request=httpx.Request("POST", "http://embedding/v1/embeddings"),
        ),
    )
    with pytest.raises(EmbeddingUnavailableError):
        provider.get_query_embedding("PRIVATE QUERY")
