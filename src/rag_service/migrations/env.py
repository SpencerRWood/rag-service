"""Resolve migration connection information from runtime settings only."""

from alembic import context
from sqlalchemy import create_engine, pool

from rag_service.config import load_settings
from rag_service.models.persistence import Base

settings = load_settings()
database_url = settings.database_url.get_secret_value()
if not database_url:
    raise ValueError("RAG_DATABASE_URL is required to run migrations")

if context.is_offline_mode():
    context.configure(
        url=database_url, target_metadata=Base.metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(database_url, poolclass=pool.NullPool, hide_parameters=True)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()
