"""Migrated database fixtures, with optional isolated PostgreSQL schemas."""

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from rag_service.dagster.launch import DagsterLauncher

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_dagster_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """API lifecycle tests reserve runs without contacting a real control plane."""
    monkeypatch.setattr(DagsterLauncher, "launch", lambda *_args: str(uuid4()))


@pytest.fixture
def migrated_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Exercise actual migrations; opt-in PostgreSQL uses a disposable schema."""
    postgres = os.environ.get("RAG_TEST_DATABASE_URL")
    admin = create_engine(postgres) if postgres else None
    schema = f"rag_test_{uuid4().hex}"
    if admin is not None:
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        url = admin.url.update_query_dict(
            {"options": f"-csearch_path={schema}"}
        ).render_as_string(hide_password=False)
    else:
        url = f"sqlite:///{tmp_path / 'rag.db'}"
    monkeypatch.setenv("RAG_DATABASE_URL", url)
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        yield url
    finally:
        if admin is not None:
            with admin.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            admin.dispose()
