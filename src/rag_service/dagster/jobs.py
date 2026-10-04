"""Secret-free runtime validation borrowed from template-python-dagster."""

from uuid import UUID

from dagster import (
    Config,
    Failure,
    OpExecutionContext,
    RetryPolicy,
    define_asset_job,
    in_process_executor,
    job,
    mem_io_manager,
    op,
)

from rag_service.dagster.resources import RAGResource
from rag_service.dagster.runtime_checks import verify_processing_runtime
from rag_service.database import create_session_factory
from rag_service.services.processing import ProcessingFailedError, process_generation

configuration_job = define_asset_job(
    "configuration_job", selection="configuration_summary"
)


@op
def runtime_smoke(context: OpExecutionContext) -> str:
    """Prove the shared runtime can execute and log a run."""
    verify_processing_runtime(context.instance)
    return "ok"


@job(executor_def=in_process_executor, resource_defs={"io_manager": mem_io_manager})
def runtime_smoke_job() -> None:
    """Validate the container independently of application resources."""
    runtime_smoke()


class ProcessingConfig(Config):
    """Only opaque durable identity enters Dagster run config/logs."""

    generation_id: str


@op(retry_policy=RetryPolicy(max_retries=2, delay=5))
def process_document(config: ProcessingConfig, rag: RAGResource) -> None:
    """Execute application transactions inside the shared Dagster runtime."""
    factory = create_session_factory(rag.settings())
    try:
        with factory() as session:
            process_generation(session, rag.storage(), UUID(config.generation_id))
    except ProcessingFailedError:
        raise Failure("Document processing failed; inspect generation status") from None
    finally:
        factory.kw["bind"].dispose()


@job(resource_defs={"rag": RAGResource()})
def ingestion_job() -> None:
    """Process an uploaded source version."""
    process_document()


@job(resource_defs={"rag": RAGResource()})
def reprocessing_job() -> None:
    """Build a new generation or retry one without re-uploading the original."""
    process_document()
