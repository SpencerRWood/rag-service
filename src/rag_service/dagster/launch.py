"""Submit reserved run identities to the infrastructure-owned GraphQL endpoint."""

import httpx

from rag_service.config import Settings
from rag_service.models.persistence import ProcessingAttempt, ProcessingGeneration

LAUNCH = """
mutation Launch($params: ExecutionParams!) {
  launchRun(executionParams: $params) {
    __typename
    ... on LaunchRunSuccess { run { runId } }
  }
}
"""
LOOKUP = """
query Run($filter: RunsFilter!) {
  runsOrError(filter: $filter, limit: 1) {
    __typename
    ... on Runs { results { runId } }
  }
}
"""


class LaunchUnavailableError(Exception):
    """The reservation remains durable and can be submitted again."""


class DagsterLauncher:
    """A bounded metadata request; never executes processing in the API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def launch(
        self, generation: ProcessingGeneration, attempt: ProcessingAttempt
    ) -> str:
        """Recover tagged submissions before requesting a new shared Dagster run."""
        try:
            with httpx.Client(timeout=10) as client:
                endpoint = self.settings.dagster_url.rstrip("/") + "/graphql"
                response = client.post(
                    endpoint,
                    json={
                        "query": LOOKUP,
                        "variables": {
                            "filter": {
                                "tags": [
                                    {"key": "rag/attempt", "value": str(attempt.id)}
                                ]
                            }
                        },
                    },
                )
                response.raise_for_status()
                lookup = response.json()["data"]["runsOrError"]
                if lookup["__typename"] != "Runs":
                    raise LaunchUnavailableError
                if lookup["results"]:
                    return str(lookup["results"][0]["runId"])
                response = client.post(
                    endpoint,
                    json={
                        "query": LAUNCH,
                        "variables": {
                            "params": {
                                "selector": {
                                    "repositoryLocationName": (
                                        self.settings.dagster_location
                                    ),
                                    "repositoryName": "__repository__",
                                    "jobName": generation.job_name,
                                },
                                "runConfigData": {
                                    "ops": {
                                        "process_document": {
                                            "config": {
                                                "generation_id": str(generation.id),
                                            }
                                        }
                                    }
                                },
                                "executionMetadata": {
                                    "tags": [
                                        {
                                            "key": "rag/generation",
                                            "value": str(generation.id),
                                        },
                                        {
                                            "key": "rag/attempt",
                                            "value": str(attempt.id),
                                        },
                                    ],
                                },
                            }
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()["data"]["launchRun"]
                if result["__typename"] != "LaunchRunSuccess":
                    raise LaunchUnavailableError
                return str(result["run"]["runId"])
        except Exception:
            raise LaunchUnavailableError from None
