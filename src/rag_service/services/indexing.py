"""Pin embedding identity and atomically publish vectors with processing state."""

from uuid import UUID

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_service.config import Settings
from rag_service.models.persistence import (
    Chunk,
    IndexedChunk,
    IndexGeneration,
    KnowledgeBase,
    ProcessingGeneration,
)
from rag_service.services.documents import DocumentConflictError
from rag_service.services.embeddings import build_embedding


def reserve_index(session: Session, knowledge_base_id: UUID) -> IndexGeneration:
    """Caller holds the KB mutation lock; implicit replacement is never allowed."""
    kb = session.get_one(KnowledgeBase, knowledge_base_id)
    index = session.scalar(
        select(IndexGeneration).where(IndexGeneration.knowledge_base_id == kb.id)
    )
    identity = (kb.embedding_provider, kb.embedding_model, kb.embedding_dimensions)
    if index is None:
        index = IndexGeneration(
            knowledge_base_id=kb.id,
            provider=identity[0],
            model=identity[1],
            dimensions=identity[2],
            endpoint=kb.embedding_endpoint,
        )
        session.add(index)
        session.flush()
    elif (index.provider, index.model, index.dimensions) != identity:
        raise DocumentConflictError(
            "Embedding configuration requires a replacement index"
        )
    return index


def index_chunks(
    session: Session,
    generation: ProcessingGeneration,
    chunks: list[Chunk],
    settings: Settings,
    embedding: BaseEmbedding | None = None,
) -> None:
    """No independent commit: ready publication includes every vector or none."""
    index = session.get_one(IndexGeneration, generation.index_generation_id)
    provider = embedding or build_embedding(index, settings)
    vectors = provider.get_text_embedding_batch([chunk.text for chunk in chunks])
    session.flush()
    for chunk, vector in zip(chunks, vectors, strict=True):
        session.add(
            IndexedChunk(
                chunk_id=chunk.id,
                processing_generation_id=generation.id,
                index_generation_id=index.id,
                provider=index.provider,
                model=index.model,
                dimensions=index.dimensions,
                embedding=vector,
            )
        )
