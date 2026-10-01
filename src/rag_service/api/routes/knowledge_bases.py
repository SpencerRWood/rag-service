"""Knowledge-base HTTP contracts backed by migrated persistent state."""

from collections.abc import Iterator
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from rag_service.config import Settings
from rag_service.models.persistence import DEFAULT_TENANT_ID, KnowledgeBase

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseCreate(BaseModel):
    """Optional overrides inherit service defaults at creation time."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    embedding_provider: Literal["local", "openrouter"] | None = None
    embedding_model: str | None = Field(default=None, min_length=1, max_length=255)
    embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_endpoint: str | None = Field(default=None, min_length=1, max_length=2048)


class KnowledgeBaseRead(BaseModel):
    """Public configuration deliberately excludes provider credentials."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    embedding_endpoint: str


def database_session(request: Request) -> Iterator[Session]:
    """Return safe API errors without disclosing database credentials or SQL."""
    factory = cast(sessionmaker[Session] | None, request.app.state.session_factory)
    if factory is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Persistence is not configured"
        )
    with factory() as session:
        try:
            yield session
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Persistence is unavailable"
            ) from exc


SessionDependency = Annotated[Session, Depends(database_session)]


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    payload: KnowledgeBaseCreate, request: Request, session: SessionDependency
) -> KnowledgeBase:
    """Persist effective configuration for the default tenant."""
    settings = cast(Settings, request.app.state.settings)
    values = {
        name: getattr(settings, name)
        for name in KnowledgeBaseCreate.model_fields
        if name != "name"
    }
    values.update(payload.model_dump(exclude_none=True))
    if values["chunk_overlap"] >= values["chunk_size"]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid effective chunk overlap"
        )
    knowledge_base = KnowledgeBase(tenant_id=DEFAULT_TENANT_ID, **values)
    session.add(knowledge_base)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        if (
            session.scalar(
                select(KnowledgeBase.id).where(
                    KnowledgeBase.tenant_id == DEFAULT_TENANT_ID,
                    KnowledgeBase.name == payload.name,
                )
            )
            is not None
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Knowledge base name already exists"
            ) from exc
        raise
    return knowledge_base


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[KnowledgeBase]:
    """Bound listing to the default tenant and stable creation order."""
    return list(
        session.scalars(
            select(KnowledgeBase)
            .where(KnowledgeBase.tenant_id == DEFAULT_TENANT_ID)
            .order_by(KnowledgeBase.created_at, KnowledgeBase.id)
            .limit(limit)
            .offset(offset)
        )
    )


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(
    knowledge_base_id: UUID, session: SessionDependency
) -> KnowledgeBase:
    """Read one configuration without crossing tenant boundaries."""
    knowledge_base = session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.tenant_id == DEFAULT_TENANT_ID,
        )
    )
    if knowledge_base is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Knowledge base not found")
    return knowledge_base
