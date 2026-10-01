"""Migration retry safety, ORM/schema agreement, and retained-data rollback."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_service.models.persistence import DEFAULT_TENANT_ID, Tenant

ROOT = Path(__file__).resolve().parents[2]


def test_migrations_are_repeatable_and_match_models(migrated_database: str) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    command.check(config)
    engine = create_engine(migrated_database)
    try:
        with Session(engine) as session:
            assert list(session.scalars(select(Tenant.name))) == ["default"]
        with pytest.raises(RuntimeError, match="without downgrading"):
            command.downgrade(config, "base")
        with Session(engine) as session:
            assert session.get(Tenant, DEFAULT_TENANT_ID) is not None
    finally:
        engine.dispose()
