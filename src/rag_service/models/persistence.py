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
