"""Reusable LlamaIndex retrieval orchestration backed by pgvector cosine search."""

import math
from uuid import UUID

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Select, cast, exists, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from rag_service.config import Settings
from rag_service.models.persistence import (
    Chunk,
    Document,
    DocumentVersion,
    IndexedChunk,
    IndexGeneration,
    ProcessingGeneration,
)
from rag_service.models.retrieval import (
    DocumentMetadata,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    SourceEvidence,
    SourceProvenance,
)
from rag_service.services.documents import DocumentNotFoundError, require_knowledge_base
from rag_service.services.embeddings import build_embedding


def source_evidence(
    chunk: Chunk, document: Document, version: DocumentVersion
) -> SourceEvidence:
    """Assemble one transport-independent, source-backed chunk."""
    return SourceEvidence(
        chunk_id=chunk.id,
        document_id=document.id,
        version_id=version.id,
        text=chunk.text,
        source=SourceProvenance(
            filename=version.filename,
            media_type=version.media_type,
            checksum=version.checksum,
            location=chunk.provenance,
        ),
        metadata=DocumentMetadata(
            title=document.title,
            tags=document.tags,
            source=document.source,
            connector_metadata=document.connector_metadata,
        ),
    )


def fetch_chunk(
    session: Session, knowledge_base_id: UUID, chunk_id: UUID
) -> SourceEvidence:
    """Read exact indexed evidence, including retained completed history."""
    require_knowledge_base(session, knowledge_base_id)
    row = session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(IndexedChunk, IndexedChunk.chunk_id == Chunk.id)
        .join(ProcessingGeneration, ProcessingGeneration.id == Chunk.generation_id)
        .join(DocumentVersion, DocumentVersion.id == ProcessingGeneration.version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            Chunk.id == chunk_id,
            Document.knowledge_base_id == knowledge_base_id,
            Document.deleted_at.is_(None),
            ProcessingGeneration.status == "ready",
            ProcessingGeneration.index_generation_id.is_not(None),
        )
    ).one_or_none()
    if row is None:
        raise DocumentNotFoundError
    return source_evidence(*row)


def candidates(
    session: Session, knowledge_base_id: UUID, request: RetrievalRequest
) -> Select[tuple[Chunk, IndexedChunk, Document, DocumentVersion]]:
    """Select latest completed generations and source versions before ranking."""
    generations = (
        select(
            ProcessingGeneration.id.label("id"),
            ProcessingGeneration.version_id.label("version_id"),
            func.row_number()
            .over(
                partition_by=ProcessingGeneration.version_id,
                order_by=ProcessingGeneration.number.desc(),
            )
            .label("rank"),
        )
        .where(
            ProcessingGeneration.status == "ready",
            ProcessingGeneration.index_generation_id.is_not(None),
        )
        .subquery()
    )
    versions = (
        select(
            DocumentVersion.id.label("id"),
            generations.c.id.label("generation_id"),
            func.row_number()
            .over(
                partition_by=DocumentVersion.document_id,
                order_by=DocumentVersion.number.desc(),
            )
            .label("rank"),
        )
        .join(generations, generations.c.version_id == DocumentVersion.id)
        .where(
            DocumentVersion.knowledge_base_id == knowledge_base_id,
            generations.c.rank == 1,
        )
        .subquery()
    )
    statement = (
        select(Chunk, IndexedChunk, Document, DocumentVersion)
        .join(IndexedChunk, IndexedChunk.chunk_id == Chunk.id)
        .join(versions, versions.c.generation_id == Chunk.generation_id)
        .join(DocumentVersion, DocumentVersion.id == versions.c.id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.deleted_at.is_(None),
        )
    )
    statement = statement.where(
        versions.c.rank == 1
        if request.version_id is None
        else DocumentVersion.id == request.version_id
    )
    for item in request.filters:
        values = [item.value] if isinstance(item.value, str) else item.value
        if item.field == "tag":
            if session.get_bind().dialect.name == "postgresql":
                predicate = or_(
                    *(cast(Document.tags, JSONB).contains([value]) for value in values)
                )
            else:
                tags = func.json_each(Document.tags).table_valued("value")
                predicate = exists(
                    select(1).select_from(tags).where(tags.c.value.in_(values))
                )
        else:
            predicate = getattr(Document, item.field).in_(values)
        statement = statement.where(predicate)
    return statement


def cosine(left: list[float], right: list[float]) -> float:
    """SQLite test adapter only; deployed retrieval uses the pgvector operator."""
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(left, right, strict=True)) / denominator),
    )


class PgVectorRetriever(BaseRetriever):
    """Read-only adapter shared by HTTP and future consumer transports."""

    def __init__(
        self,
        session: Session,
        knowledge_base_id: UUID,
        request: RetrievalRequest,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.session = session
        self.knowledge_base_id = knowledge_base_id
        self.request = request
        self.settings = settings

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        index = self.session.scalar(
            select(IndexGeneration).where(
                IndexGeneration.knowledge_base_id == self.knowledge_base_id
            )
        )
        if index is None:
            return []
        statement = candidates(self.session, self.knowledge_base_id, self.request)
        if self.session.execute(statement.limit(1)).first() is None:
            return []
        vector = build_embedding(index, self.settings).get_query_embedding(
            query_bundle.query_str
        )
        if self.session.get_bind().dialect.name == "postgresql":
            distance = cast(IndexedChunk.embedding, Vector()).cosine_distance(vector)
            rows = self.session.execute(
                statement.add_columns(cast(1 - distance, Float))
                .order_by(distance, Chunk.id)
                .limit(self.request.result_count)
            ).all()
        else:
            scored = [
                (*row, cosine(row[1].embedding, vector))
                for row in self.session.execute(statement)
            ]
            rows = sorted(scored, key=lambda row: (-row[4], str(row[0].id)))[
                : self.request.result_count
            ]  # type: ignore[assignment]
        nodes = []
        for chunk, _indexed, document, version, score in rows:
            result = RetrievalResult(
                **source_evidence(chunk, document, version).model_dump(),
                score=max(-1.0, min(1.0, score)),
            )
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        id_=str(chunk.id),
                        text=chunk.text,
                        metadata={"result": result.model_dump(mode="json")},
                    ),
                    score=result.score,
                )
            )
        return nodes


def retrieve(
    session: Session,
    knowledge_base_id: UUID,
    request: RetrievalRequest,
    settings: Settings,
) -> RetrievalResponse:
    """Validate ownership even for empty indexes; explicit history stays KB scoped."""
    require_knowledge_base(session, knowledge_base_id)
    if (
        request.version_id is not None
        and session.scalar(
            select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                DocumentVersion.id == request.version_id,
                Document.knowledge_base_id == knowledge_base_id,
                Document.deleted_at.is_(None),
            )
        )
        is None
    ):
        raise DocumentNotFoundError
    nodes = PgVectorRetriever(session, knowledge_base_id, request, settings).retrieve(
        request.query
    )
    return RetrievalResponse(
        knowledge_base_id=knowledge_base_id,
        results=[
            RetrievalResult.model_validate(node.node.metadata["result"])
            for node in nodes
        ],
    )
