"""Durable tenant and knowledge-base configuration records."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint
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
