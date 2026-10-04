"""Exercise the real GraphQL launch schema against a managed RAG code location."""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from dagster import DagsterInstance, DagsterRunStatus
from dagster._core.workspace.context import WorkspaceProcessContext
from dagster._core.workspace.load_target import ModuleTarget
from dagster_graphql.schema import create_schema

from rag_service.config import Settings
from rag_service.dagster.launch import DagsterLauncher, LaunchUnavailableError
from rag_service.models.persistence import ProcessingAttempt, ProcessingGeneration

REAL_CLIENT = httpx.Client
REAL_LAUNCH = DagsterLauncher.launch


def test_graphql_submission_and_lost_response_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = create_schema()
    home = tmp_path / "dagster"
    home.mkdir()
    (home / "dagster.yaml").write_text(
        "run_coordinator:\n  module: dagster._core.run_coordinator\n"
        "  class: QueuedRunCoordinator\ntelemetry: {enabled: false}\n"
    )
    generation = ProcessingGeneration(
        id=uuid4(),
        version_id=uuid4(),
        operation_key="ingest",
        number=1,
        job_name="ingestion_job",
        chunk_size=10,
        chunk_overlap=2,
    )
    attempt = ProcessingAttempt(
        id=uuid4(), generation_id=generation.id, request_key="initial"
    )
    monkeypatch.setattr(DagsterLauncher, "launch", REAL_LAUNCH)
    with (
        DagsterInstance.local_temp(str(home)) as instance,
        WorkspaceProcessContext(
            instance,
            ModuleTarget(
                "rag_service.dagster.definitions", "defs", None, "rag-service"
            ),
        ) as workspace,
    ):
        context = workspace.create_request_context()
        submitted = 0
        lose_response = True

        def graphql(request: httpx.Request) -> httpx.Response:
            nonlocal submitted, lose_response
            payload = json.loads(request.content)
            result = asyncio.run(
                schema.execute_async(
                    payload["query"],
                    context_value=context,
                    variable_values=payload["variables"],
                )
            )
            assert not result.errors
            if "params" in payload["variables"]:
                submitted += 1
                assert result.data is not None
                assert result.data["launchRun"]["__typename"] == "LaunchRunSuccess"
                if lose_response:
                    lose_response = False
                    raise httpx.ReadTimeout("response lost")
            return httpx.Response(200, json={"data": result.data})

        monkeypatch.setattr(
            "rag_service.dagster.launch.httpx.Client",
            lambda **_kwargs: REAL_CLIENT(transport=httpx.MockTransport(graphql)),
        )
        launcher = DagsterLauncher(Settings())
        with pytest.raises(LaunchUnavailableError):
            launcher.launch(generation, attempt)
        run_id = launcher.launch(generation, attempt)
        assert launcher.launch(generation, attempt) == run_id
        assert submitted == 1
        run = instance.get_run_by_id(run_id)
        assert run is not None
        assert run.status == DagsterRunStatus.QUEUED
        assert run.tags["rag/generation"] == str(generation.id)
        assert run.tags["rag/attempt"] == str(attempt.id)


@pytest.mark.parametrize(
    "response",
    [
        {"errors": [{"message": "private internal error"}]},
        {"data": {"runsOrError": {"__typename": "PythonError"}}},
        {"data": {"launchRun": {"__typename": "RunConfigValidationInvalid"}}},
    ],
)
def test_launch_failures_are_safe(
    response: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if "launchRun" in response.get("data", {}) and b"params" not in request.content:
            return httpx.Response(
                200,
                json={"data": {"runsOrError": {"__typename": "Runs", "results": []}}},
            )
        return httpx.Response(200, json=response)

    monkeypatch.setattr(DagsterLauncher, "launch", REAL_LAUNCH)
    monkeypatch.setattr(
        "rag_service.dagster.launch.httpx.Client",
        lambda **_kwargs: REAL_CLIENT(transport=httpx.MockTransport(transport)),
    )
    generation = ProcessingGeneration(id=uuid4(), job_name="ingestion_job")
    attempt = ProcessingAttempt(id=uuid4())
    with pytest.raises(LaunchUnavailableError) as failure:
        DagsterLauncher(Settings()).launch(generation, attempt)
    assert "private" not in str(failure.value)
