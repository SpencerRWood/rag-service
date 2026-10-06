"""Bounded runtime configuration using the existing RAG embedding settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL: Literal["Qwen/Qwen3-Embedding-0.6B"] = "Qwen/Qwen3-Embedding-0.6B"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", hide_input_in_errors=True)

    embedding_model: Literal["Qwen/Qwen3-Embedding-0.6B"] = MODEL
    embedding_dimensions: int = Field(default=1024, ge=1024, le=1024)
    embedding_timeout: float = Field(default=25, gt=0, le=300)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_cache_path: Path = Path("/data/models")
    embedding_model_revision: str = Field(
        default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        min_length=40,
        max_length=40,
        pattern="^[0-9a-f]+$",
    )
    embedding_startup_timeout: float = Field(default=600, gt=0, le=1800)
    embedding_max_tokens: int = Field(default=8192, ge=1, le=32768)
    embedding_threads: int = Field(default=4, ge=1, le=64)
    source_revision: str = "unknown"
    release_revision: str = "unknown"
