"""Durable tenant and knowledge-base configuration records."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


class Base(DeclarativeBase):
    """Metadata exported to Alembic."""


class Tenant(Base):
    """Tenant identity reserved for future isolation/authentication work."""

    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class KnowledgeBase(Base):
    """Effective settings are snapshotted when a knowledge base is created."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_knowledge_base_tenant_name"),
        CheckConstraint("chunk_size > 0", name="ck_chunk_size_positive"),
        CheckConstraint(
            "chunk_overlap >= 0 AND chunk_overlap < chunk_size",
            name="ck_chunk_overlap_valid",
        ),
        CheckConstraint(
            "embedding_dimensions > 0", name="ck_embedding_dimensions_positive"
        ),
        CheckConstraint(
            "embedding_provider IN ('local', 'openrouter')",
            name="ck_embedding_provider",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    embedding_provider: Mapped[str] = mapped_column(String(32))
    embedding_model: Mapped[str] = mapped_column(String(255))
    embedding_dimensions: Mapped[int]
    embedding_endpoint: Mapped[str] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Document(Base):
    """Stable knowledge-base-scoped identity, independent of filename/content."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("id", "knowledge_base_id", name="uq_document_scope"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id"), index=True
    )
    title: Mapped[str] = mapped_column(String(255))
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    source: Mapped[str | None] = mapped_column(String(2048))
    connector_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentVersion(Base):
    """Immutable source identity with retry-safe original-storage/lifecycle state."""

    __tablename__ = "document_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_source_document_scope",
        ),
        UniqueConstraint("knowledge_base_id", "checksum", name="uq_source_checksum"),
        UniqueConstraint("document_id", "number", name="uq_source_version_number"),
        CheckConstraint("number > 0", name="ck_source_version_number"),
        CheckConstraint("size >= 0", name="ck_source_size"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_source_status",
        ),
        CheckConstraint(
            "status = 'failed' OR original_stored", name="ck_source_original_stored"
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_bases.id"))
    document_id: Mapped[UUID] = mapped_column(index=True)
    number: Mapped[int]
    checksum: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(255))
    size: Mapped[int]
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    original_stored: Mapped[bool]
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ParsedContent(Base):
    """Normalized parser output, reusable independently of derived chunks."""

    __tablename__ = "parsed_contents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"))
    parser_name: Mapped[str] = mapped_column(String(64))
    parser_version: Mapped[str] = mapped_column(String(64))
    segments: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    parsed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ProcessingGeneration(Base):
    """Idempotent operation identity and immutable effective chunking settings."""

    __tablename__ = "processing_generations"
    __table_args__ = (
        UniqueConstraint("version_id", "operation_key", name="uq_processing_operation"),
        UniqueConstraint("version_id", "number", name="uq_processing_number"),
        CheckConstraint("number > 0", name="ck_processing_number"),
        CheckConstraint("chunk_size > 0", name="ck_processing_chunk_size"),
        CheckConstraint(
            "chunk_overlap >= 0 AND chunk_overlap < chunk_size",
            name="ck_processing_overlap",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_processing_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("document_versions.id"))
    operation_key: Mapped[str] = mapped_column(String(255))
    number: Mapped[int]
    job_name: Mapped[str] = mapped_column(String(64))
    reparse: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    parsed_content_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("parsed_contents.id")
    )
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProcessingAttempt(Base):
    """Durable launch reservation survives ambiguous network outcomes/restarts."""

    __tablename__ = "processing_attempts"
    __table_args__ = (
        UniqueConstraint("generation_id", "request_key", name="uq_attempt_request"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generation_id: Mapped[UUID] = mapped_column(ForeignKey("processing_generations.id"))
    request_key: Mapped[str] = mapped_column(String(255))
    submitted: Mapped[bool] = mapped_column(default=False)
    dagster_run_id: Mapped[str | None] = mapped_column(String(255))


class Chunk(Base):
    """Ordered, provenance-rich derived text belonging to one generation."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("generation_id", "ordinal", name="uq_chunk_ordinal"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generation_id: Mapped[UUID] = mapped_column(ForeignKey("processing_generations.id"))
    ordinal: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    provenance: Mapped[dict[str, object]] = mapped_column(JSON)
