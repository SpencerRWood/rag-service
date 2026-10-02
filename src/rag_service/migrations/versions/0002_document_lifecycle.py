"""Persist source identities, originals, metadata, and lifecycle history."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Keep deduplication and version sequencing durable across restarts."""
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(2048), nullable=True),
        sa.Column("connector_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("id", "knowledge_base_id", name="uq_document_scope"),
    )
    op.create_index(
        "ix_documents_knowledge_base_id", "documents", ["knowledge_base_id"]
    )
    op.create_table(
        "document_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id"),
            nullable=False,
        ),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(255), nullable=False, unique=True),
        sa.Column("original_stored", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_source_document_scope",
        ),
        sa.UniqueConstraint("knowledge_base_id", "checksum", name="uq_source_checksum"),
        sa.UniqueConstraint("document_id", "number", name="uq_source_version_number"),
        sa.CheckConstraint("number > 0", name="ck_source_version_number"),
        sa.CheckConstraint("size >= 0", name="ck_source_size"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_source_status",
        ),
        sa.CheckConstraint(
            "status = 'failed' OR original_stored", name="ck_source_original_stored"
        ),
    )
    op.create_index(
        "ix_document_versions_document_id", "document_versions", ["document_id"]
    )


def downgrade() -> None:
    """Rollback application images while retaining source history."""
    raise RuntimeError(
        "Restore the previous application image without downgrading the database"
    )
