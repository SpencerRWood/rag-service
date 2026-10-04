"""Importable code-location entrypoint matching template-python-dagster."""

from dagster import Definitions

from rag_service.dagster.assets import configuration_summary
from rag_service.dagster.jobs import (
    configuration_job,
    ingestion_job,
    reprocessing_job,
    runtime_smoke_job,
)
from rag_service.dagster.resources import RAGResource
from rag_service.dagster.sensors import recover_launches

defs = Definitions(
    assets=[configuration_summary],
    jobs=[configuration_job, runtime_smoke_job, ingestion_job, reprocessing_job],
    sensors=[recover_launches],
    resources={"rag": RAGResource()},
)
