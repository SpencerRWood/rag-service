"""Durable processing reservations and transactional publication for Dagster."""

from datetime import UTC, datetime
from time import perf_counter
from typing import cast
from uuid import UUID

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rag_service.config import Settings, load_settings
from rag_service.dagster.launch import DagsterLauncher, LaunchUnavailableError
from rag_service.models.persistence import (
    Chunk,
    DocumentVersion,
    KnowledgeBase,
    ParsedContent,
    ProcessingAttempt,
    ProcessingGeneration,
    ProcessingObservation,
)
from rag_service.services.documents import (
    DocumentConflictError,
    find_document,
    find_version,
    lock_knowledge_base,
)
from rag_service.services.indexing import index_chunks, reserve_index
from rag_service.services.parsing import parse
from rag_service.storage import FileStorage


def reserve_generation(  # noqa: PLR0913
    session: Session,
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    operation_key: str,
    *,
    reparse: bool = False,
) -> tuple[ProcessingGeneration, ProcessingAttempt]:
    """Serialize duplicate launches; reserve before making any network request."""
    lock_knowledge_base(session, knowledge_base_id)
    find_document(session, knowledge_base_id, document_id)
    source = find_version(session, document_id, version_id)
    if not source.original_stored:
        raise DocumentConflictError("Original must be stored before processing")
    generation = session.scalar(
        select(ProcessingGeneration).where(
            ProcessingGeneration.version_id == version_id,
            ProcessingGeneration.operation_key == operation_key,
        )
    )
    if generation is None:
        kb = session.get_one(KnowledgeBase, knowledge_base_id)
        last = session.scalar(
            select(ProcessingGeneration.number)
            .where(ProcessingGeneration.version_id == version_id)
            .order_by(ProcessingGeneration.number.desc())
            .limit(1)
        )
        parsed = (
            None
            if reparse
            else session.scalar(
                select(ParsedContent.id)
                .where(ParsedContent.version_id == version_id)
                .order_by(ParsedContent.parsed_at.desc())
                .limit(1)
            )
        )
        generation = ProcessingGeneration(
            index_generation_id=reserve_index(session, knowledge_base_id).id,
            version_id=version_id,
            operation_key=operation_key,
            number=(last or 0) + 1,
            job_name="ingestion_job"
            if operation_key == "ingest"
            else "reprocessing_job",
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
            reparse=reparse,
            parsed_content_id=parsed,
        )
        session.add(generation)
        session.flush()
    elif generation.reparse != reparse:
        raise DocumentConflictError(
            "Idempotency key already used with different options"
        )
    attempt = session.scalar(
        select(ProcessingAttempt).where(
            ProcessingAttempt.generation_id == generation.id,
            ProcessingAttempt.request_key == "initial",
        )
    )
    if attempt is None:
        attempt = ProcessingAttempt(generation_id=generation.id, request_key="initial")
        session.add(attempt)
    session.commit()
    return generation, attempt


def retry_generation(  # noqa: PLR0913, PLR0917
    session: Session,
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    generation_id: UUID,
    request_key: str,
) -> tuple[ProcessingGeneration, ProcessingAttempt]:
    """Retries keep operation/chunk identity; a request key selects one run."""
    lock_knowledge_base(session, knowledge_base_id)
    find_document(session, knowledge_base_id, document_id)
    find_version(session, document_id, version_id)
    generation = session.scalar(
        select(ProcessingGeneration).where(
            ProcessingGeneration.id == generation_id,
            ProcessingGeneration.version_id == version_id,
        )
    )
    if generation is None:
        raise DocumentConflictError("Unknown processing generation")
    attempt = session.scalar(
        select(ProcessingAttempt).where(
            ProcessingAttempt.generation_id == generation_id,
            ProcessingAttempt.request_key == request_key,
        )
    )
    if attempt is None:
        if generation.status != "failed":
            raise DocumentConflictError("Only failed processing can be retried")
        generation.status = "pending"
        generation.error_code = None
        attempt = ProcessingAttempt(
            generation_id=generation_id, request_key=request_key
        )
        session.add(attempt)
    session.commit()
    return generation, attempt


def process_generation(  # noqa: PLR0915 -- two crash-safe transaction stages
    session: Session,
    storage: FileStorage,
    generation_id: UUID,
    settings: Settings | None = None,
    embedding: BaseEmbedding | None = None,
) -> None:
    """Dagster-only work; crash-safe parse checkpoint and atomic chunk publication.

    The knowledge-base mutation lock spans each stage. Interrupted stages roll
    back; retries see either the parse checkpoint or a fully completed generation.
    No partially written chunks are visible as ready. Source ready state is never
    cleared by a failed reprocessing generation.
    """
    generation = session.get_one(ProcessingGeneration, generation_id)
    source = session.get_one(DocumentVersion, generation.version_id)
    kb_id, doc_id = source.knowledge_base_id, source.document_id
    session.rollback()
    started = perf_counter()
    embedding_duration: float | None = None
    try:
        lock_knowledge_base(session, kb_id)
        find_document(session, kb_id, doc_id)
        session.refresh(generation)
        if generation.status == "ready":
            session.commit()
            return
        if generation.index_generation_id is None:
            generation.index_generation_id = reserve_index(session, kb_id).id
        generation.status = "processing"
        if source.status != "ready":
            source.status = "processing"
        if generation.parsed_content_id is None:
            name, parser_version, segments = parse(
                storage.read(source.storage_key), source.filename
            )
            parsed = ParsedContent(
                version_id=source.id,
                parser_name=name,
                parser_version=parser_version,
                segments=segments,
            )
            session.add(parsed)
            session.flush()
            generation.parsed_content_id = parsed.id
        session.commit()
        lock_knowledge_base(session, kb_id)
        find_document(session, kb_id, doc_id)
        session.refresh(generation)
        if generation.status == "ready":
            session.commit()
            return
        parsed = session.get_one(ParsedContent, generation.parsed_content_id)
        splitter = SentenceSplitter(
            chunk_size=generation.chunk_size,
            chunk_overlap=generation.chunk_overlap,
            tokenizer=lambda text: text.split(),
        )
        chunks: list[Chunk] = []
        for part in parsed.segments:
            provenance = dict(cast(dict[str, object], part["provenance"]))
            header = str(provenance.get("table_header", ""))
            for value in splitter.split_text(str(part["text"])):
                chunk = Chunk(
                    generation_id=generation.id,
                    ordinal=len(chunks),
                    text=f"{header}\n{value}" if header else value,
                    provenance=provenance,
                )
                session.add(chunk)
                chunks.append(chunk)
        if not chunks:
            raise ValueError("No extractable content")
        embedding_started = perf_counter()
        try:
            index_chunks(
                session, generation, chunks, settings or load_settings(), embedding
            )
        finally:
            embedding_duration = perf_counter() - embedding_started
        generation.status = "ready"
        generation.error_code = None
        generation.completed_at = datetime.now(UTC)
        source.status = "ready"
        session.add(
            ProcessingObservation(
                outcome="success",
                duration_seconds=perf_counter() - started,
                embedding_duration_seconds=embedding_duration,
                chunk_count=len(chunks),
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        lock_knowledge_base(session, kb_id)
        session.refresh(generation)
        session.refresh(source)
        if generation.status != "ready":
            generation.status = "failed"
            generation.error_code = "processing_failed"
            if source.status != "ready":
                source.status = "failed"
            session.add(
                ProcessingObservation(
                    outcome="failure",
                    duration_seconds=perf_counter() - started,
                    embedding_duration_seconds=embedding_duration,
                    chunk_count=0,
                )
            )
        session.commit()
        raise ProcessingFailedError from None


class ProcessingFailedError(Exception):
    """Safe failure boundary: parser/storage exception text stays out of Dagster."""


def submit_processing(
    session: Session,
    settings: Settings,
    generation: ProcessingGeneration,
    attempt: ProcessingAttempt,
) -> None:
    """Serialize launches and recover lost responses through a durable attempt tag."""
    session.execute(
        update(ProcessingAttempt)
        .where(ProcessingAttempt.id == attempt.id)
        .values(submitted=ProcessingAttempt.submitted)
    )
    session.refresh(attempt)
    if attempt.submitted:
        session.commit()
        return
    try:
        attempt.dagster_run_id = DagsterLauncher(settings).launch(generation, attempt)
    except LaunchUnavailableError:
        session.rollback()
        return
    attempt.submitted = True
    session.commit()
