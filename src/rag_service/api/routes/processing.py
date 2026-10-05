"""Explicit source reprocessing, failed-generation retries and durable history."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from rag_service.api.routes.documents import (
    DocumentSession,
    UTCDatetime,
)
from rag_service.models.persistence import ProcessingAttempt, ProcessingGeneration
from rag_service.services.documents import (
    find_document,
    find_version,
    require_knowledge_base,
)
from rag_service.services.processing import (
    reserve_generation,
    retry_generation,
    submit_processing,
)

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/documents/{document_id}/versions/{version_id}",
    tags=["processing"],
)
IdempotencyKey = Annotated[
    str, Header(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")
]


class ReprocessRequest(BaseModel):
    """Re-chunk stored parsed content by default; reparsing is explicit."""

    model_config = ConfigDict(extra="forbid")
    reparse: bool = False


class GenerationRead(BaseModel):
    """New ready generations include indexed chunks; legacy identity is nullable."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    index_generation_id: UUID | None
    version_id: UUID
    number: int
    status: str
    chunk_size: int
    chunk_overlap: int
    parsed_content_id: UUID | None
    error_code: str | None
    created_at: UTCDatetime
    completed_at: UTCDatetime | None


class LaunchRead(BaseModel):
    """Opaque IDs link the application operation to shared Dagster run storage."""

    generation: GenerationRead
    dagster_run_id: str | None
    launch_submitted: bool


@router.post("/reprocess", response_model=LaunchRead, status_code=202)
def reprocess(  # noqa: PLR0913, PLR0917
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    payload: ReprocessRequest,
    request: Request,
    session: DocumentSession,
    idempotency_key: IdempotencyKey,
) -> LaunchRead:
    """A new caller operation key creates a generation, never a source version."""
    generation, attempt = reserve_generation(
        session,
        knowledge_base_id,
        document_id,
        version_id,
        f"reprocess:{idempotency_key}",
        reparse=payload.reparse,
    )
    submit_processing(session, request.app.state.settings, generation, attempt)
    return launch_read(generation, attempt)


@router.post(
    "/generations/{generation_id}/retry", response_model=LaunchRead, status_code=202
)
def retry(  # noqa: PLR0913, PLR0917
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    generation_id: UUID,
    request: Request,
    session: DocumentSession,
    idempotency_key: IdempotencyKey,
) -> LaunchRead:
    """Explicitly reserve another Dagster run for failed processing."""
    generation, attempt = retry_generation(
        session,
        knowledge_base_id,
        document_id,
        version_id,
        generation_id,
        f"retry:{idempotency_key}",
    )
    submit_processing(session, request.app.state.settings, generation, attempt)
    return launch_read(generation, attempt)


def launch_read(
    generation: ProcessingGeneration, attempt: ProcessingAttempt
) -> LaunchRead:
    """Shared response shape for reprocess and retry."""
    return LaunchRead(
        generation=GenerationRead.model_validate(generation),
        dagster_run_id=attempt.dagster_run_id,
        launch_submitted=attempt.submitted,
    )


@router.get("/generations", response_model=list[GenerationRead])
def generations(  # noqa: PLR0913, PLR0917
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    session: DocumentSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ProcessingGeneration]:
    """Durable status, parse checkpoint and effective settings survive API restart."""
    require_knowledge_base(session, knowledge_base_id)
    find_document(session, knowledge_base_id, document_id)
    find_version(session, document_id, version_id)
    return list(
        session.scalars(
            select(ProcessingGeneration)
            .where(
                ProcessingGeneration.version_id == version_id,
            )
            .order_by(ProcessingGeneration.number.desc())
            .limit(limit)
            .offset(offset)
        )
    )
