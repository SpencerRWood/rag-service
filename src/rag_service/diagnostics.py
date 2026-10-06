"""Bounded dependency probes return states, never settings or exception text."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryFile

import boto3
import httpx
from botocore.config import Config
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from rag_service.config import Settings
from rag_service.models.persistence import KnowledgeBase, ProcessingObservation

LOCATION_QUERY = """
query { repositoriesOrError { __typename
  ... on RepositoryConnection { nodes { location { name } } }
} }
"""


def probe(operation: Callable[[], None]) -> str:
    """Even credential-bearing provider failures remain a fixed safe state."""
    try:
        operation()
    except Exception:
        return "unavailable"
    return "available"


def database_health(factory: sessionmaker[Session]) -> None:
    """Check connectivity and required migrated tables, without scanning content."""
    with factory() as session:
        if session.get_bind().dialect.name == "postgresql":
            session.execute(text("SET LOCAL statement_timeout = 2000"))
        session.execute(select(KnowledgeBase.id).limit(1))
        session.execute(select(ProcessingObservation.id).limit(1))


def storage_health(settings: Settings) -> None:
    """Verify local write access or configured S3 bucket connectivity."""
    if settings.storage_backend == "filesystem":
        with TemporaryFile(dir=settings.storage_path) as file:
            file.write(b"rag-readiness")
            file.flush()
    else:
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            region_name=settings.s3_region,
            config=Config(
                connect_timeout=2, read_timeout=2, retries={"max_attempts": 0}
            ),
        )
        try:
            client.head_bucket(Bucket=settings.s3_bucket)
        finally:
            client.close()


def dagster_health(settings: Settings) -> None:
    """Require this code location to be loaded, rather than just an HTTP listener."""
    with httpx.Client(timeout=2, follow_redirects=False) as client:
        response = client.post(
            settings.dagster_url.rstrip("/") + "/graphql",
            json={"query": LOCATION_QUERY},
        )
        response.raise_for_status()
        result = response.json()["data"]["repositoriesOrError"]
    if result["__typename"] != "RepositoryConnection" or not any(
        node["location"]["name"] == settings.dagster_location
        for node in result["nodes"]
    ):
        raise ValueError("Code location is unavailable")


def embedding_health(settings: Settings) -> None:
    """Read the model catalog without sending documents or generating text."""
    key = settings.embedding_api_key.get_secret_value()
    if settings.embedding_provider == "openrouter" and not key:
        raise ValueError("Embedding credential is missing")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    catalog = (
        "/embeddings/models"
        if settings.embedding_provider == "openrouter"
        else "/models"
    )
    with httpx.Client(timeout=2, follow_redirects=False) as client:
        response = client.get(
            settings.embedding_endpoint.rstrip("/") + catalog, headers=headers
        )
        response.raise_for_status()
        models = response.json()["data"]
    if not any(model["id"] == settings.embedding_model for model in models):
        raise ValueError("Configured model is unavailable")


def dependencies(
    settings: Settings, factory: sessionmaker[Session] | None
) -> dict[str, str]:
    """Database/storage gate readiness; optional integrations remain observable."""
    operations = {
        "storage": lambda: storage_health(settings),
        "dagster": lambda: dagster_health(settings),
        "embedding": lambda: embedding_health(settings),
    }
    if factory is not None:
        operations["database"] = lambda: database_health(factory)
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {
            name: executor.submit(probe, operation)
            for name, operation in operations.items()
        }
        states = {name: future.result() for name, future in pending.items()}
    if factory is None:
        states["database"] = "not_configured"
    return states
