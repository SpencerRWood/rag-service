"""Stable retrieval contract independent of vector storage internals."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EqualityFilter(BaseModel):
    """Exact, case-sensitive metadata equality."""

    model_config = ConfigDict(extra="forbid")
    field: Literal["title", "source", "tag"]
    operator: Literal["eq"]
    value: str = Field(max_length=2048)


class MembershipFilter(BaseModel):
    """IN matches any supplied value; tag matches an individual document tag."""

    model_config = ConfigDict(extra="forbid")
    field: Literal["title", "source", "tag"]
    operator: Literal["in"]
    value: list[Annotated[str, Field(max_length=2048)]] = Field(
        min_length=1, max_length=100
    )


MetadataFilter = Annotated[
    EqualityFilter | MembershipFilter, Field(discriminator="operator")
]


class RetrievalRequest(BaseModel):
    """Knowledge base comes from the resource path; filters combine with AND."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=8192)
    result_count: int = Field(default=5, ge=1, le=100)
    version_id: UUID | None = None
    filters: list[MetadataFilter] = Field(default_factory=list, max_length=20)


class SourceProvenance(BaseModel):
    """Immutable source locator and parser-preserved location information."""

    filename: str
    media_type: str
    checksum: str
    location: dict[str, object]


class DocumentMetadata(BaseModel):
    """Current descriptive metadata, shared by filtering and result rendering."""

    title: str
    tags: list[str]
    source: str | None
    connector_metadata: dict[str, object]


class SourceEvidence(BaseModel):
    """Stable chunk identity and source context shared by search and fetch."""

    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    text: str
    source: SourceProvenance
    metadata: DocumentMetadata


class RetrievalResult(SourceEvidence):
    """Ranked evidence; cosine similarity is in [-1, 1], larger is better."""

    score: float


class RetrievalResponse(BaseModel):
    """The public response deliberately omits database/provider configuration."""

    knowledge_base_id: UUID
    results: list[RetrievalResult]
