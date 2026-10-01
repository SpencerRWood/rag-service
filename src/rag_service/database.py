"""SQLAlchemy persistence setup; schema changes belong to Alembic."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag_service.config import Settings


def create_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Build a lazy engine without connecting or migrating at startup."""
    url = settings.database_url.get_secret_value()
    if not url:
        raise ValueError("RAG_DATABASE_URL is required for persistence")
    engine = create_engine(url, pool_pre_ping=True, hide_parameters=True)
    return sessionmaker(engine, expire_on_commit=False)
