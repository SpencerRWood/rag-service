"""API lifecycle tests inject an encoder; image smoke uses the real Qwen model."""

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_embedding.config import MODEL, Settings
from rag_embedding.main import create_app


class FakeEncoder:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.vectors: list[list[float]] | None = None
        self.failure = False
        self.gate: threading.Event | None = None

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.inputs = texts
        if self.gate:
            self.gate.wait(2)
        if self.failure:
            raise RuntimeError("private input and credentials")
        return self.vectors or [[1.0] + [0.0] * 1023 for _ in texts], len(texts)


def wait_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/ready").status_code == 200:
            return
        time.sleep(0.01)
    pytest.fail("model did not load")


@contextmanager
def runtime(encoder: FakeEncoder, **settings: object) -> Iterator[TestClient]:
    config = Settings.model_validate(settings)
    with TestClient(create_app(config, lambda _: encoder)) as client:
        wait_ready(client)
        yield client


@pytest.mark.parametrize(
    "settings",
    [
        {"embedding_model": "other"},
        {"embedding_dimensions": 512},
        {"embedding_timeout": 0},
        {"embedding_batch_size": 0},
        {"embedding_model_revision": "main"},
        {"embedding_threads": 0},
        {"embedding_max_tokens": 32769},
        {"embedding_startup_timeout": 0},
    ],
)
def test_invalid_configuration(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(settings)


def test_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_TIMEOUT", "12")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "8")
    config = Settings()
    assert config.embedding_model == MODEL
    assert config.embedding_dimensions == 1024
    assert config.embedding_timeout == 12
    assert config.embedding_batch_size == 8


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"input": []},
        {"input": " "},
        {"input": 3},
        {"input": [3]},
        {"input": ["ok", ""]},
        {"input": ["x"] * 257},
        {"input": "x" * 32769},
        {"input": ["x" * 32768] * 9},
        {"input": "ok", "dimensions": 512},
        {"input": "ok", "model": "other"},
        {"input": "ok", "encoding_format": "base64"},
        {"input": "ok", "extra": True},
    ],
)
def test_request_validation(payload: dict[str, object]) -> None:
    with runtime(FakeEncoder()) as client:
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert "input" not in response.json()["error"]


@pytest.mark.parametrize("inputs", ["document", ["first", "second"]])
def test_response_shape(inputs: str | list[str]) -> None:
    encoder = FakeEncoder()
    with runtime(encoder) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": inputs,
                "model": MODEL,
                "dimensions": 1024,
                "encoding_format": "float",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == MODEL
        assert data["object"] == "list"
        texts = [inputs] if isinstance(inputs, str) else inputs
        assert encoder.inputs == texts
        assert len(data["data"]) == len(texts)
        assert data["usage"]["total_tokens"] == len(texts)
        for index, item in enumerate(data["data"]):
            assert item["index"] == index and item["object"] == "embedding"
            assert len(item["embedding"]) == 1024
            assert all(math.isfinite(value) for value in item["embedding"])


def test_load_once_and_health_during_startup() -> None:
    gate = threading.Event()
    calls = []

    def loader(_: Settings) -> FakeEncoder:
        calls.append(1)
        gate.wait(2)
        return FakeEncoder()

    with TestClient(create_app(Settings(), loader)) as client:
        try:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/ready").status_code == 503
            assert client.get("/v1/models").status_code == 503
            assert (
                client.post("/v1/embeddings", json={"input": "ok"}).status_code == 503
            )
        finally:
            gate.set()
        wait_ready(client)
        assert client.get("/v1/models").json()["data"][0]["id"] == MODEL
        for _ in range(2):
            assert (
                client.post("/v1/embeddings", json={"input": "ok"}).status_code == 200
            )
        assert len(calls) == 1


def test_loading_failure(caplog: pytest.LogCaptureFixture) -> None:
    def fail(_: Settings) -> FakeEncoder:
        raise RuntimeError("private credential")

    with TestClient(create_app(Settings(), fail)) as client:
        for _ in range(100):
            if client.get("/ready").json()["status"] == "failed":
                break
            time.sleep(0.01)
        assert client.get("/ready").status_code == 503
        assert client.get("/ready").json()["status"] == "failed"
        assert client.get("/health").status_code == 200
        assert client.post("/v1/embeddings", json={"input": "ok"}).status_code == 503
    assert "model_load_failed" in caplog.text
    assert "private credential" not in caplog.text


def test_startup_timeout() -> None:
    gate = threading.Event()

    def slow(_: Settings) -> FakeEncoder:
        gate.wait(2)
        return FakeEncoder()

    with TestClient(
        create_app(Settings(embedding_startup_timeout=0.01), slow)
    ) as client:
        try:
            for _ in range(100):
                if client.get("/ready").json()["status"] == "failed":
                    break
                time.sleep(0.01)
            assert client.get("/ready").json()["status"] == "failed"
            assert client.get("/health").status_code == 200
        finally:
            gate.set()


def test_inference_timeout_retains_slot() -> None:
    encoder = FakeEncoder()
    encoder.gate = threading.Event()
    with runtime(encoder, embedding_timeout=0.02) as client:
        try:
            assert (
                client.post("/v1/embeddings", json={"input": "ok"}).status_code == 504
            )
            assert client.get("/health").status_code == 200
            assert (
                client.post("/v1/embeddings", json={"input": "ok"}).status_code == 429
            )
        finally:
            encoder.gate.set()


@pytest.mark.parametrize(
    "vectors",
    [[[1.0]], [[float("nan")] * 1024], [[0.0] * 1024], [[1.0] * 1024, [1.0] * 1024]],
)
def test_invalid_model_output(vectors: list[list[float]]) -> None:
    encoder = FakeEncoder()
    encoder.vectors = vectors
    with runtime(encoder) as client:
        response = client.post("/v1/embeddings", json={"input": "ok"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_failed"


def test_inference_failure(caplog: pytest.LogCaptureFixture) -> None:
    encoder = FakeEncoder()
    encoder.failure = True
    with runtime(encoder) as client:
        assert client.post("/v1/embeddings", json={"input": "ok"}).status_code == 500
    assert "private input" not in caplog.text


def test_body_limit_and_invalid_json() -> None:
    with runtime(FakeEncoder()) as client:
        assert client.post("/v1/embeddings", content=b"x" * 1048577).status_code == 413
        assert client.post("/v1/embeddings", content="{").status_code == 422
