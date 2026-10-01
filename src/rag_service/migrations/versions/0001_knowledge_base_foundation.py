"""Create tenant and knowledge-base configuration without source/index tables."""

from uuid import UUID

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Version the persistence contract and seed the default tenant."""
    tenants = op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
    )
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("embedding_provider", sa.String(32), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding_endpoint", sa.String(2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("tenant_id", "name", name="uq_knowledge_base_tenant_name"),
        sa.CheckConstraint("chunk_size > 0", name="ck_chunk_size_positive"),
        sa.CheckConstraint(
            "chunk_overlap >= 0 AND chunk_overlap < chunk_size",
            name="ck_chunk_overlap_valid",
        ),
        sa.CheckConstraint(
            "embedding_dimensions > 0", name="ck_embedding_dimensions_positive"
        ),
        sa.CheckConstraint(
            "embedding_provider IN ('local', 'openrouter')",
            name="ck_embedding_provider",
        ),
    )
    op.create_index("ix_knowledge_bases_tenant_id", "knowledge_bases", ["tenant_id"])
    op.bulk_insert(
        tenants,
        [{"id": UUID("00000000-0000-0000-0000-000000000001"), "name": "default"}],
    )


def downgrade() -> None:
    """Application rollback retains durable data; destructive downgrade is refused."""
    raise RuntimeError(
        "Restore the previous application image without downgrading the database"
    )
