"""Add content-free worker observations without changing source history."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Persist cross-process measurements in an additive table."""
    op.create_table(
        "processing_observations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("embedding_duration_seconds", sa.Float()),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "outcome IN ('success', 'failure')", name="ck_observation_outcome"
        ),
        sa.CheckConstraint("duration_seconds >= 0", name="ck_observation_duration"),
        sa.CheckConstraint("chunk_count >= 0", name="ck_observation_chunks"),
    )


def downgrade() -> None:
    """Image rollback retains the schema and durable observations."""
    raise RuntimeError("Restore the previous image without downgrading the database")
