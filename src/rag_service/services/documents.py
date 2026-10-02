"""Transactional source lifecycle shared by HTTP and future processing workers."""

from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rag_service.models.persistence import (
    DEFAULT_TENANT_ID,
    Document,
    DocumentVersion,
    KnowledgeBase,
)
from rag_service.storage import FileStorage

VersionStatus = Literal["pending", "processing", "ready", "failed"]


class DocumentNotFoundError(Exception):
    """The requested identity is absent or outside the caller's scope."""


class DocumentConflictError(Exception):
    """Content belongs to another identity, or a transition is invalid."""


class OriginalUnavailableError(Exception):
    """Original storage failed; retained identifiers allow safe retries."""


def require_knowledge_base(session: Session, knowledge_base_id: UUID) -> None:
    """Scope reads to the current tenant."""
    if (
        session.scalar(
            select(KnowledgeBase.id).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.tenant_id == DEFAULT_TENANT_ID,
            )
        )
        is None
    ):
        raise DocumentNotFoundError


def lock_knowledge_base(session: Session, knowledge_base_id: UUID) -> None:
    """Serialize mutations with a row write on both PostgreSQL and SQLite.

    This must be the transaction's first statement. Unlike SELECT FOR UPDATE,
    UPDATE also acquires SQLite's write lock before deduplication reads. The lock
    spans original storage and commit, so delete/upload/worker transitions cannot
    race. Database constraints remain the final uniqueness authority.
    """
    result = session.execute(
        update(KnowledgeBase)
        .where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.tenant_id == DEFAULT_TENANT_ID,
        )
        .values(name=KnowledgeBase.name)
        .returning(KnowledgeBase.id)
    )
    if result.scalar_one_or_none() is None:
        raise DocumentNotFoundError


def find_document(
    session: Session,
    knowledge_base_id: UUID,
    document_id: UUID,
    *,
    include_deleted: bool = False,
) -> Document:
    """Explicit history reads may include retained soft-deleted records."""
    statement = select(Document).where(
        Document.knowledge_base_id == knowledge_base_id, Document.id == document_id
    )
    if not include_deleted:
        statement = statement.where(Document.deleted_at.is_(None))
    document = session.scalar(statement)
    if document is None:
        raise DocumentNotFoundError
    return document


def find_version(
    session: Session, document_id: UUID, version_id: UUID
) -> DocumentVersion:
    """Resolve version identity through its owning document."""
    version = session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.id == version_id,
        )
    )
    if version is None:
        raise DocumentNotFoundError
    return version


def source_versions(session: Session, document_id: UUID) -> list[DocumentVersion]:
    """Order newest first; ready selection never depends on completion order."""
    return list(
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.number.desc())
        )
    )


def upload_original(  # noqa: PLR0913, PLR0917 -- explicit source/metadata boundary
    session: Session,
    storage: FileStorage,
    knowledge_base_id: UUID,
    content: bytes,
    filename: str,
    media_type: str,
    *,
    document_id: UUID | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
    connector_metadata: dict[str, object] | None = None,
) -> tuple[Document, DocumentVersion, bool]:
    """Reserve identity and store exact bytes under one serialized transaction.

    A deterministic content key makes writes retry-safe even if a process dies
    after storage succeeds but before the database commit. Storage failures keep
    a failed version for retry, without changing any older ready source. Retrying
    an already stored version never resets processing state or metadata.
    """
    lock_knowledge_base(session, knowledge_base_id)
    document = (
        find_document(session, knowledge_base_id, document_id)
        if document_id is not None
        else None
    )
    checksum = sha256(content).hexdigest()
    existing = session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.knowledge_base_id == knowledge_base_id,
            DocumentVersion.checksum == checksum,
        )
    )
    created = existing is None
    if existing is not None:
        owner = find_document(
            session, knowledge_base_id, existing.document_id, include_deleted=True
        )
        if owner.deleted_at is not None:
            raise DocumentConflictError("Content belongs to a deleted document")
        if document is not None and document.id != owner.id:
            raise DocumentConflictError("Content already belongs to another document")
        document, version = owner, existing
    else:
        if document is None:
            document = Document(
                knowledge_base_id=knowledge_base_id,
                title=title or filename,
                tags=tags or [],
                source=source,
                connector_metadata=connector_metadata or {},
            )
            session.add(document)
            session.flush()
        latest_number = session.scalar(
            select(DocumentVersion.number)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.number.desc())
            .limit(1)
        )
        version = DocumentVersion(
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            number=(latest_number or 0) + 1,
            checksum=checksum,
            filename=filename,
            media_type=media_type,
            size=len(content),
            storage_key=f"originals/{knowledge_base_id}/{checksum}",
            original_stored=False,
            status="failed",
        )
        session.add(version)
    if not version.original_stored:
        try:
            storage.save(version.storage_key, content)
        except Exception as exc:
            # Adapter failures must not expose source contents/provider secrets.
            session.commit()
            raise OriginalUnavailableError from exc
        version.original_stored = True
        version.status = "pending"
    session.commit()
    return document, version, created


def transition_version(
    session: Session,
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    target: VersionStatus,
) -> DocumentVersion:
    """Internal worker boundary; HTTP callers cannot declare sources ready.

    Repeated transitions are no-ops. Ready/failed outcomes require processing;
    retrying a failed processing job explicitly transitions back to processing.
    The active source is the highest numbered ready version, so late completion
    of an older version cannot replace a newer ready source.
    """
    lock_knowledge_base(session, knowledge_base_id)
    find_document(session, knowledge_base_id, document_id)
    version = find_version(session, document_id, version_id)
    allowed = {
        "pending": {"processing"},
        "processing": {"ready", "failed"},
        "ready": set(),
        "failed": {"processing"},
    }
    if not version.original_stored or (
        target != version.status and target not in allowed[version.status]
    ):
        raise DocumentConflictError("Invalid source-version transition")
    version.status = target
    session.commit()
    return version


def soft_delete(session: Session, knowledge_base_id: UUID, document_id: UUID) -> None:
    """Retain history/checksum reservations while excluding normal use."""
    lock_knowledge_base(session, knowledge_base_id)
    document = find_document(
        session, knowledge_base_id, document_id, include_deleted=True
    )
    if document.deleted_at is None:
        document.deleted_at = datetime.now(UTC)
    session.commit()
