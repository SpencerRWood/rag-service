"""Code-location loadability and resource-free runtime execution."""

from dagster import Definitions

from rag_service.dagster.definitions import defs
from rag_service.dagster.resources import RAGResource
from rag_service.storage import FilesystemStorage


def test_definitions_and_smoke_run() -> None:
    Definitions.validate_loadable(defs)
    result = defs.get_job_def("runtime_smoke_job").execute_in_process()
    assert result.success
    assert result.output_for_node("runtime_smoke") == "ok"
    configured = defs.get_job_def("configuration_job").execute_in_process()
    assert configured.success
    assert configured.output_for_node("configuration_summary") == {
        "embedding_provider": "local",
        "embedding_model": "Qwen3-Embedding-0.6B",
        "embedding_dimensions": 1024,
        "chunk_size": 512,
        "chunk_overlap": 64,
    }


def test_typed_resource_shares_runtime_storage() -> None:
    resource = RAGResource()
    assert resource.settings().embedding_model == "Qwen3-Embedding-0.6B"
    assert isinstance(resource.storage(), FilesystemStorage)
