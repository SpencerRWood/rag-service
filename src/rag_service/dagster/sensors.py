"""Recover durable launch reservations interrupted before API submission."""

from uuid import UUID

from dagster import (
    DagsterRunStatus,
    DefaultSensorStatus,
    SensorEvaluationContext,
    sensor,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_service.dagster.jobs import ingestion_job, reprocessing_job
from rag_service.dagster.resources import RAGResource
from rag_service.database import create_session_factory
from rag_service.models.persistence import (
    Document,
    DocumentVersion,
    ProcessingAttempt,
    ProcessingGeneration,
)
from rag_service.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    lock_knowledge_base,
)
from rag_service.services.processing import reserve_generation, submit_processing


@sensor(
    jobs=[ingestion_job, reprocessing_job],
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.RUNNING,
)
def recover_launches(context: SensorEvaluationContext, rag: RAGResource) -> None:
    """Bounded outbox recovery uses the same explicit reserved GraphQL run IDs."""
    factory = create_session_factory(rag.settings())
    try:
        with factory() as session:
            sources = list(
                session.execute(
                    select(
                        DocumentVersion.knowledge_base_id,
                        DocumentVersion.document_id,
                        DocumentVersion.id,
                    )
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        Document.deleted_at.is_(None),
                        DocumentVersion.original_stored.is_(True),
                        DocumentVersion.status == "pending",
                        ~select(ProcessingGeneration.id)
                        .where(
                            ProcessingGeneration.version_id == DocumentVersion.id,
                            ProcessingGeneration.operation_key == "ingest",
                        )
                        .exists(),
                    )
                    .order_by(DocumentVersion.created_at)
                    .limit(50)
                )
            )
            session.rollback()
            for kb_id, doc_id, source_id in sources:
                try:
                    reserve_generation(session, kb_id, doc_id, source_id, "ingest")
                except DocumentConflictError, DocumentNotFoundError:
                    session.rollback()
            attempts = list(
                session.scalars(
                    select(ProcessingAttempt)
                    .where(ProcessingAttempt.submitted.is_(False))
                    .order_by(ProcessingAttempt.id)
                    .limit(50)
                )
            )
            for attempt in attempts:
                generation = session.get_one(
                    ProcessingGeneration, attempt.generation_id
                )
                submit_processing(session, rag.settings(), generation, attempt)
            generations = list(
                session.scalars(
                    select(ProcessingGeneration)
                    .where(ProcessingGeneration.status.in_(["pending", "processing"]))
                    .order_by(ProcessingGeneration.created_at)
                    .limit(50)
                )
            )
            for generation in generations:
                attempts = list(
                    session.scalars(
                        select(ProcessingAttempt).where(
                            ProcessingAttempt.generation_id == generation.id
                        )
                    )
                )
                runs = [
                    context.instance.get_run_by_id(attempt.dagster_run_id)
                    for attempt in attempts
                    if attempt.dagster_run_id
                ]
                if (
                    attempts
                    and all(attempt.submitted for attempt in attempts)
                    and len(runs) == len(attempts)
                    and all(
                        run is not None
                        and run.status
                        in {DagsterRunStatus.FAILURE, DagsterRunStatus.CANCELED}
                        for run in runs
                    )
                ):
                    mark_interrupted(
                        session, generation.id, {attempt.id for attempt in attempts}
                    )
    finally:
        factory.kw["bind"].dispose()


def mark_interrupted(
    session: Session, generation_id: UUID, attempt_ids: set[UUID]
) -> None:
    """A terminal failed Dagster run makes interrupted application work retryable."""
    generation = session.get_one(ProcessingGeneration, generation_id)
    source = session.get_one(DocumentVersion, generation.version_id)
    kb_id = source.knowledge_base_id
    session.rollback()
    lock_knowledge_base(session, kb_id)
    session.refresh(generation)
    session.refresh(source)
    current_ids = set(
        session.scalars(
            select(ProcessingAttempt.id).where(
                ProcessingAttempt.generation_id == generation_id
            )
        )
    )
    if current_ids != attempt_ids:
        session.commit()
        return
    if generation.status in {"pending", "processing"}:
        generation.status = "failed"
        generation.error_code = "dagster_run_failed"
        if source.status != "ready":
            source.status = "failed"
    session.commit()
