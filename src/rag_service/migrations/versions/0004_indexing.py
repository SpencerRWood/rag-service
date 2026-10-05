"""Add embedding spaces and durable pgvector embeddings without rewriting history."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Legacy ready generations remain preserved; explicit reprocessing indexes them."""
    postgres = op.get_bind().dialect.name == "postgresql"
    if postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    op.create_table(
        "index_generations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("endpoint", sa.String(2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("knowledge_base_id", name="uq_index_knowledge_base"),
        sa.UniqueConstraint(
            "id", "provider", "model", "dimensions", name="uq_index_identity"
        ),
        sa.CheckConstraint("dimensions > 0", name="ck_index_dimensions"),
        sa.CheckConstraint(
            "provider IN ('local', 'openrouter')", name="ck_index_provider"
        ),
    )
    with op.batch_alter_table("processing_generations") as batch:
        batch.add_column(sa.Column("index_generation_id", sa.Uuid()))
        batch.create_foreign_key(
            "fk_processing_index", "index_generations", ["index_generation_id"], ["id"]
        )
        batch.create_unique_constraint(
            "uq_processing_index", ["id", "index_generation_id"]
        )
    op.create_table(
        "indexed_chunks",
        sa.Column("chunk_id", sa.Uuid(), sa.ForeignKey("chunks.id"), primary_key=True),
        sa.Column("processing_generation_id", sa.Uuid(), nullable=False),
        sa.Column("index_generation_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column(
            "embedding", Vector().with_variant(sa.JSON(), "sqlite"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["index_generation_id", "provider", "model", "dimensions"],
            [
                "index_generations.id",
                "index_generations.provider",
                "index_generations.model",
                "index_generations.dimensions",
            ],
            name="fk_indexed_identity",
        ),
        sa.ForeignKeyConstraint(
            ["processing_generation_id", "index_generation_id"],
            ["processing_generations.id", "processing_generations.index_generation_id"],
            name="fk_indexed_processing",
        ),
    )
    if postgres:
        op.create_check_constraint(
            "ck_indexed_vector_dimensions",
            "indexed_chunks",
            "vector_dims(embedding) = dimensions",
        )


def downgrade() -> None:
    """Retain source/index history on image rollback."""
    raise RuntimeError("Restore the previous image without downgrading the database")
