"""Resource-backed foundation asset, following template-python-dagster."""

from dagster import AssetExecutionContext, PythonObjectDagsterType, asset

from rag_service.dagster.resources import RAGResource


@asset(dagster_type=PythonObjectDagsterType(dict, name="ConfigurationSummary"))
def configuration_summary(
    context: AssetExecutionContext, rag: RAGResource
) -> dict[str, str | int]:
    """Expose non-secret configuration without executing ingestion or indexing."""
    settings = rag.settings()
    context.log.info("RAG configuration resolved")
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
