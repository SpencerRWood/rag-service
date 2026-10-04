"""Application fixture executed inside the centralized candidate-image gate."""

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dagster import DagsterInstance
from dagster_postgres import PostgresRunStorage
from pydantic import SecretStr
from sqlalchemy import create_engine, make_url, select, text

from rag_service.config import Settings
from rag_service.database import create_session_factory
from rag_service.models.persistence import (
    DEFAULT_TENANT_ID,
    Base,
    Chunk,
    KnowledgeBase,
    ParsedContent,
    Tenant,
)
from rag_service.services.documents import upload_original
from rag_service.services.processing import process_generation, reserve_generation
from rag_service.storage import build_storage


def verify_processing_runtime(instance: DagsterInstance) -> None:
    """Prove candidate dependencies and transactions using disposable fixture state.

    The shared gate supplies its PostgreSQL credentials. Without gate credentials,
    local smoke tests use temporary SQLite. This fixture never reads the real RAG
    database, documents, or provider credentials and needs no model inference.
    """
    schema = f"rag_smoke_{uuid4().hex}"
    admin = None
    with TemporaryDirectory(prefix="rag-runtime-") as directory:
        database_url = f"sqlite:///{directory}/fixture.db"
        if isinstance(instance.run_storage, PostgresRunStorage):
            url = make_url(instance.run_storage.postgres_url)
            admin = create_engine(url, hide_parameters=True)
            with admin.begin() as connection:
                connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            database_url = url.update_query_dict(
                {"options": f"-csearch_path={schema}"}
            ).render_as_string(hide_password=False)
        settings = Settings(
            database_url=SecretStr(database_url),
            storage_path=Path(directory) / "originals",
        )
        factory = create_session_factory(settings)
        try:
            Base.metadata.create_all(factory.kw["bind"])
            with factory() as session:
                session.add(Tenant(id=DEFAULT_TENANT_ID, name="default"))
                session.flush()
                kb = KnowledgeBase(
                    tenant_id=DEFAULT_TENANT_ID,
                    name="fixture",
                    chunk_size=8,
                    chunk_overlap=2,
                    embedding_provider="local",
                    embedding_model="fixture",
                    embedding_dimensions=1,
                    embedding_endpoint="http://unused",
                )
                session.add(kb)
                session.commit()
                storage = build_storage(settings)
                document, source, _ = upload_original(
                    session,
                    storage,
                    kb.id,
                    b"Name,Value\nAlpha,1\nBeta,2",
                    "fixture.csv",
                    "text/csv",
                )
                generation, _ = reserve_generation(
                    session, kb.id, document.id, source.id, "ingest"
                )
                process_generation(session, storage, generation.id)
                process_generation(session, storage, generation.id)
                chunks = list(session.scalars(select(Chunk)))
                parsed = list(session.scalars(select(ParsedContent)))
                if (
                    generation.status != "ready"
                    or len(chunks) != 2
                    or len(parsed) != 1
                    or chunks[0].provenance["row_start"] != 2
                ):
                    raise RuntimeError("Candidate processing fixture failed")
        finally:
            factory.kw["bind"].dispose()
            if admin is not None:
                with admin.begin() as connection:
                    connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                admin.dispose()
