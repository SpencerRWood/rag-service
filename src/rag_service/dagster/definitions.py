"""Importable code-location entrypoint matching template-python-dagster."""

from dagster import Definitions

from rag_service.dagster.assets import configuration_summary
from rag_service.dagster.jobs import configuration_job, runtime_smoke_job
from rag_service.dagster.resources import RAGResource

defs = Definitions(
    assets=[configuration_summary],
    jobs=[configuration_job, runtime_smoke_job],
    resources={"rag": RAGResource()},
)
