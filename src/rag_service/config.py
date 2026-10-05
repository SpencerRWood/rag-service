"""Validated application settings; deployment injects runtime secrets."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shared API and code-location settings without tracking credentials."""

    model_config = SettingsConfigDict(env_prefix="RAG_", hide_input_in_errors=True)

    environment: str = "development"
    database_url: SecretStr = SecretStr("")
    storage_backend: Literal["filesystem", "s3"] = "filesystem"
    storage_path: Path = Path("./data/documents")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    s3_bucket: str = ""
    s3_endpoint: str | None = None
    s3_region: str = "us-east-1"
    embedding_provider: Literal["local", "openrouter"] = "local"
    embedding_model: str = Field(default="Qwen3-Embedding-0.6B", min_length=1)
    embedding_dimensions: int = Field(default=1024, gt=0)
    embedding_endpoint: str = "http://localhost:8080/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_timeout: float = Field(default=30, gt=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    dagster_url: str = "http://localhost:3000"
    dagster_location: str = "rag-service"
    source_revision: str = "unknown"
    release_revision: str = "unknown"
    mcp_path: str = "/mcp"
    mcp_allowed_hosts: list[str] = ["localhost:*", "127.0.0.1:*", "[::1]:*"]
    mcp_allowed_origins: list[str] = [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://[::1]:*",
    ]

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Reject invalid resource and chunking choices before serving requests."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("s3_bucket is required for S3 storage")
        if (
            not self.mcp_path.startswith("/")
            or self.mcp_path == "/"
            or self.mcp_path.endswith("/")
            or any(character in self.mcp_path for character in "{}?#")
        ):
            raise ValueError(
                "mcp_path must be an absolute non-root route without suffix"
            )
        return self


def load_settings() -> Settings:
    """Load environment variables without requiring or reading a secret file."""
    return Settings()
