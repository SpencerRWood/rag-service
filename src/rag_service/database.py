"""SQLAlchemy persistence setup; schema changes belong to Alembic."""

from sqlite3 import Connection

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from rag_service.config import Settings


def create_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Build a lazy engine without connecting or migrating at startup."""
    url = settings.database_url.get_secret_value()
    if not url:
        raise ValueError("RAG_DATABASE_URL is required for persistence")
    engine = create_engine(url, pool_pre_ping=True, hide_parameters=True)

    @event.listens_for(engine, "connect")
    def enforce_sqlite_foreign_keys(connection: object, _record: object) -> None:
        if isinstance(connection, Connection):
            connection.execute("PRAGMA foreign_keys=ON")

    return sessionmaker(engine, expire_on_commit=False)
