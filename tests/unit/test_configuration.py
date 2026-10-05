"""Application defaults, runtime overrides, and secret handling."""

import pytest
from pydantic import SecretStr, ValidationError

from rag_service.config import Settings, load_settings
from rag_service.database import create_session_factory


def test_runtime_overrides_and_secret_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "provider/model")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "test-provider-key")
    monkeypatch.setenv(
        "RAG_DATABASE_URL", "postgresql+psycopg2://user:test-password@db/rag"
    )
    settings = load_settings()
    assert settings.embedding_provider == "openrouter"
    assert settings.embedding_model == "provider/model"
    assert settings.embedding_dimensions == 768
    assert "test-provider-key" not in repr(settings)
    assert "test-password" not in repr(settings)
    assert "OPENPROJECT" not in repr(settings)


@pytest.mark.parametrize("path", ["mcp", "/", "/mcp/", "/{path}", "/mcp?x=1"])
def test_invalid_mcp_path_is_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="mcp_path"):
        Settings(mcp_path=path)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=32)
    with pytest.raises(ValidationError, match="s3_bucket"):
        Settings(storage_backend="s3")
    with pytest.raises(ValueError, match="RAG_DATABASE_URL"):
        create_session_factory(Settings(database_url=SecretStr("")))
