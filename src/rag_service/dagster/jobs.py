"""Secret-free runtime validation borrowed from template-python-dagster."""

from dagster import define_asset_job, in_process_executor, job, mem_io_manager, op

configuration_job = define_asset_job(
    "configuration_job", selection="configuration_summary"
)


@op
def runtime_smoke() -> str:
    """Prove the shared runtime can execute and log a run."""
    return "ok"


@job(executor_def=in_process_executor, resource_defs={"io_manager": mem_io_manager})
def runtime_smoke_job() -> None:
    """Validate the container independently of application resources."""
    runtime_smoke()
