"""Persist normalized content, processing operations, launch attempts and chunks."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the processing contract without rewriting retained source state."""
    op.create_table(
        "parsed_contents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "version_id",
            sa.Uuid(),
            sa.ForeignKey("document_versions.id"),
            nullable=False,
        ),
        sa.Column("parser_name", sa.String(64), nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False),
        sa.Column("segments", sa.JSON(), nullable=False),
        sa.Column("parsed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "processing_generations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "version_id",
            sa.Uuid(),
            sa.ForeignKey("document_versions.id"),
            nullable=False,
        ),
        sa.Column("operation_key", sa.String(255), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("job_name", sa.String(64), nullable=False),
        sa.Column("reparse", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("parsed_content_id", sa.Uuid(), sa.ForeignKey("parsed_contents.id")),
        sa.Column("error_code", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "version_id", "operation_key", name="uq_processing_operation"
        ),
        sa.UniqueConstraint("version_id", "number", name="uq_processing_number"),
        sa.CheckConstraint("number > 0", name="ck_processing_number"),
        sa.CheckConstraint("chunk_size > 0", name="ck_processing_chunk_size"),
        sa.CheckConstraint(
            "chunk_overlap >= 0 AND chunk_overlap < chunk_size",
            name="ck_processing_overlap",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_processing_status",
        ),
    )
    op.create_table(
        "processing_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("processing_generations.id"),
            nullable=False,
        ),
        sa.Column("request_key", sa.String(255), nullable=False),
        sa.Column("submitted", sa.Boolean(), nullable=False),
        sa.Column("dagster_run_id", sa.String(255)),
        sa.UniqueConstraint("generation_id", "request_key", name="uq_attempt_request"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("processing_generations.id"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.UniqueConstraint("generation_id", "ordinal", name="uq_chunk_ordinal"),
    )


def downgrade() -> None:
    """Image rollback retains durable processing history."""
    raise RuntimeError("Restore the previous image without downgrading the database")
