"""Knowledge-base-scoped original uploads, source history, and metadata."""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated, cast
from urllib.parse import quote
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_service.api.routes.knowledge_bases import SessionDependency
from rag_service.config import Settings
from rag_service.models.persistence import (
    Document,
    DocumentVersion,
    ProcessingGeneration,
)
from rag_service.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    OriginalUnavailableError,
    find_document,
    find_version,
    lock_knowledge_base,
    require_knowledge_base,
    soft_delete,
    upload_original,
)
from rag_service.services.processing import reserve_generation, submit_processing
from rag_service.storage import FileStorage, build_storage

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/documents", tags=["documents"]
)


def utc_datetime(value: datetime) -> datetime:
    """SQLite drops tzinfo; persisted service timestamps are always UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(utc_datetime)]


class DocumentMetadata(BaseModel):
    """Creation defaults; exact retries preserve the previously saved metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=255)
    tags: list[Annotated[str, Field(min_length=1, max_length=255)]] = Field(
        default_factory=list, max_length=100
    )
    source: str | None = Field(default=None, max_length=2048)
    connector_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class VersionRead(BaseModel):
    """Expose source identity and lifecycle without disclosing storage placement."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    number: int
    checksum: str
    filename: str
    media_type: str
    size: int
    original_stored: bool
    status: str
    created_at: UTCDatetime


class DocumentRead(DocumentMetadata):
    """Latest attempted and active successfully ready versions are separate."""

    id: UUID
    knowledge_base_id: UUID
    created_at: UTCDatetime
    deleted_at: UTCDatetime | None
    status: str
    latest_version_id: UUID
    active_version_id: UUID | None
    active_processing_generation_id: UUID | None


class UploadRead(BaseModel):
    """Exact duplicate uploads return the original document and version IDs."""

    document: DocumentRead
    version: VersionRead
    created: bool
    generation_id: UUID
    dagster_run_id: str | None
    launch_submitted: bool


def document_session(session: SessionDependency) -> Iterator[Session]:
    """Map scoped lifecycle errors without exposing provider details."""
    try:
        yield session
    except DocumentNotFoundError as exc:
        session.rollback()
        raise HTTPException(
            404, "Document, version, or knowledge base not found"
        ) from exc
    except DocumentConflictError as exc:
        session.rollback()
        raise HTTPException(409, str(exc)) from exc
    except OriginalUnavailableError as exc:
        session.rollback()
        raise HTTPException(503, "Original storage is unavailable") from exc


DocumentSession = Annotated[Session, Depends(document_session)]


def original_storage(request: Request) -> FileStorage:
    """Resolve on demand so liveness never depends on storage connectivity."""
    try:
        return build_storage(cast(Settings, request.app.state.settings))
    except Exception as exc:
        raise HTTPException(503, "Original storage is unavailable") from exc


StorageDependency = Annotated[FileStorage, Depends(original_storage)]


def document_read(session: Session, document: Document) -> DocumentRead:
    """Active means indexed; legacy parsed-only history is retained but inactive."""
    versions = select(DocumentVersion).where(DocumentVersion.document_id == document.id)
    latest = session.scalars(
        versions.order_by(DocumentVersion.number.desc()).limit(1)
    ).one()
    active = session.scalar(
        select(DocumentVersion.id)
        .join(
            ProcessingGeneration, ProcessingGeneration.version_id == DocumentVersion.id
        )
        .where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.status == "ready",
            ProcessingGeneration.status == "ready",
            ProcessingGeneration.index_generation_id.is_not(None),
        )
        .order_by(DocumentVersion.number.desc())
        .limit(1)
    )
    return DocumentRead(
        id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        title=document.title,
        tags=document.tags,
        source=document.source,
        connector_metadata=cast(dict[str, JsonValue], document.connector_metadata),
        created_at=document.created_at,
        deleted_at=document.deleted_at,
        status="deleted" if document.deleted_at is not None else latest.status,
        latest_version_id=latest.id,
        active_version_id=active,
        active_processing_generation_id=session.scalar(
            select(ProcessingGeneration.id)
            .where(
                ProcessingGeneration.version_id == active,
                ProcessingGeneration.status == "ready",
                ProcessingGeneration.index_generation_id.is_not(None),
            )
            .order_by(ProcessingGeneration.number.desc())
            .limit(1)
        )
        if active
        else None,
    )


def receive_original(file: UploadFile, settings: Settings) -> tuple[bytes, str, str]:
    """Bound memory and reject invalid public filename/content-type fields."""
    filename = (file.filename or "original").replace("\\", "/").rsplit("/", 1)[-1]
    media_type = file.content_type or "application/octet-stream"
    if (
        not filename
        or len(filename) > 255
        or len(media_type) > 255
        or not media_type.isascii()
        or any(ord(char) < 32 or ord(char) == 127 for char in filename + media_type)
    ):
        raise HTTPException(422, "Invalid original filename or media type")
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "Original exceeds the configured upload limit")
    return content, filename, media_type


@router.post("", response_model=UploadRead, status_code=201)
def create_document(  # noqa: PLR0913, PLR0917 -- HTTP path/form/dependency parameters
    knowledge_base_id: UUID,
    request: Request,
    response: Response,
    session: DocumentSession,
    storage: StorageDependency,
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()] = "{}",
) -> UploadRead:
    """Multipart exact-original upload with optional JSON metadata form field."""
    try:
        fields = DocumentMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(422, "Invalid document metadata") from exc
    content, filename, media_type = receive_original(file, request.app.state.settings)
    document, version, created = upload_original(
        session,
        storage,
        knowledge_base_id,
        content,
        filename,
        media_type,
        title=fields.title,
        tags=fields.tags,
        source=fields.source,
        connector_metadata=dict(fields.connector_metadata),
    )
    if not created:
        response.status_code = status.HTTP_200_OK
    generation, attempt = reserve_generation(
        session, knowledge_base_id, document.id, version.id, "ingest"
    )
    submit_processing(session, request.app.state.settings, generation, attempt)
    return UploadRead(
        document=document_read(session, document),
        version=VersionRead.model_validate(version),
        created=created,
        generation_id=generation.id,
        dagster_run_id=attempt.dagster_run_id,
        launch_submitted=attempt.submitted,
    )


@router.post("/{document_id}/versions", response_model=UploadRead, status_code=201)
def create_version(  # noqa: PLR0913, PLR0917 -- HTTP path/form/dependency parameters
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    response: Response,
    session: DocumentSession,
    storage: StorageDependency,
    file: Annotated[UploadFile, File()],
) -> UploadRead:
    """Changed bytes create a source version; identical bytes retain their ID."""
    content, filename, media_type = receive_original(file, request.app.state.settings)
    document, version, created = upload_original(
        session,
        storage,
        knowledge_base_id,
        content,
        filename,
        media_type,
        document_id=document_id,
    )
    if not created:
        response.status_code = status.HTTP_200_OK
    generation, attempt = reserve_generation(
        session, knowledge_base_id, document.id, version.id, "ingest"
    )
    submit_processing(session, request.app.state.settings, generation, attempt)
    return UploadRead(
        document=document_read(session, document),
        version=VersionRead.model_validate(version),
        created=created,
        generation_id=generation.id,
        dagster_run_id=attempt.dagster_run_id,
        launch_submitted=attempt.submitted,
    )


@router.get("", response_model=list[DocumentRead])
def list_documents(
    knowledge_base_id: UUID,
    session: DocumentSession,
    include_deleted: bool = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentRead]:
    """Soft-deleted identities are visible only by explicit history opt-in."""
    require_knowledge_base(session, knowledge_base_id)
    statement = select(Document).where(Document.knowledge_base_id == knowledge_base_id)
    if not include_deleted:
        statement = statement.where(Document.deleted_at.is_(None))
    return [
        document_read(session, document)
        for document in session.scalars(
            statement.order_by(Document.created_at, Document.id)
            .limit(limit)
            .offset(offset)
        )
    ]


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    session: DocumentSession,
    include_deleted: bool = False,
) -> DocumentRead:
    """Read persisted metadata and active/latest source IDs."""
    require_knowledge_base(session, knowledge_base_id)
    return document_read(
        session,
        find_document(
            session, knowledge_base_id, document_id, include_deleted=include_deleted
        ),
    )


@router.patch("/{document_id}", response_model=DocumentRead)
def update_metadata(
    knowledge_base_id: UUID,
    document_id: UUID,
    payload: DocumentMetadata,
    session: DocumentSession,
) -> DocumentRead:
    """Update explicitly supplied metadata without rewriting source history."""
    if "title" in payload.model_fields_set and payload.title is None:
        raise HTTPException(422, "Document title cannot be null")
    lock_knowledge_base(session, knowledge_base_id)
    document = find_document(session, knowledge_base_id, document_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(document, field, value)
    session.commit()
    return document_read(session, document)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    session: DocumentSession,
) -> Response:
    """Retry-safe soft deletion never removes originals or version history."""
    soft_delete(session, knowledge_base_id, document_id)
    return Response(status_code=204)


@router.get("/{document_id}/versions", response_model=list[VersionRead])
def list_versions(  # noqa: PLR0913, PLR0917 -- scoped history and pagination
    knowledge_base_id: UUID,
    document_id: UUID,
    session: DocumentSession,
    include_deleted: bool = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentVersion]:
    """Retain all historical source versions, newest first."""
    require_knowledge_base(session, knowledge_base_id)
    find_document(
        session, knowledge_base_id, document_id, include_deleted=include_deleted
    )
    return list(
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.number.desc())
            .limit(limit)
            .offset(offset)
        )
    )


@router.get("/{document_id}/versions/{version_id}", response_model=VersionRead)
def get_version(
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    session: DocumentSession,
    include_deleted: bool = False,
) -> DocumentVersion:
    """Read one immutable source identity and its current lifecycle state."""
    require_knowledge_base(session, knowledge_base_id)
    find_document(
        session, knowledge_base_id, document_id, include_deleted=include_deleted
    )
    return find_version(session, document_id, version_id)


@router.get("/{document_id}/versions/{version_id}/original")
def download_original(  # noqa: PLR0913, PLR0917 -- scoped history and storage
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    session: DocumentSession,
    storage: StorageDependency,
    include_deleted: bool = False,
) -> Response:
    """Return exact bytes for a source version, including opted-in history."""
    version = get_version(
        knowledge_base_id, document_id, version_id, session, include_deleted
    )
    if not version.original_stored:
        raise OriginalUnavailableError
    try:
        content = storage.read(version.storage_key)
    except Exception as exc:
        raise OriginalUnavailableError from exc
    return Response(
        content,
        media_type=version.media_type,
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''"
            + quote(version.filename, safe="")
        },
    )
