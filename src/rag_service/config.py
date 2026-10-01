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
    s3_bucket: str = ""
    s3_endpoint: str | None = None
    s3_region: str = "us-east-1"
    embedding_provider: Literal["local", "openrouter"] = "local"
    embedding_model: str = Field(default="Qwen3-Embedding-0.6B", min_length=1)
    embedding_dimensions: int = Field(default=1024, gt=0)
    embedding_endpoint: str = "http://localhost:8080"
    embedding_api_key: SecretStr = SecretStr("")
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    dagster_url: str = "http://localhost:3000"
    source_revision: str = "unknown"
    release_revision: str = "unknown"

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        """Reject invalid resource and chunking choices before serving requests."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("s3_bucket is required for S3 storage")
        return self


def load_settings() -> Settings:
    """Load environment variables without requiring or reading a secret file."""
    return Settings()
