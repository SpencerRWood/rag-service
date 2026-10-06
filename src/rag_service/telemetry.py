"""Bounded, content-free Prometheus measurements shared by HTTP and MCP."""

from collections.abc import Callable, Iterator
from functools import wraps
from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import CounterMetricFamily, Metric, SummaryMetricFamily
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from rag_service.models.persistence import ProcessingObservation
from rag_service.models.retrieval import RetrievalResponse

UPLOADS = Counter(
    "rag_upload_requests", "Original upload operations", ["outcome"], registry=None
)
RETRIEVALS = Counter(
    "rag_retrieval_requests", "HTTP and MCP searches", ["outcome"], registry=None
)
LATENCY = Histogram("rag_retrieval_duration_seconds", "Search duration", registry=None)
QUERY_EMBEDDING = Histogram(
    "rag_query_embedding_duration_seconds", "Query embedding duration", registry=None
)
RESULTS = Histogram(
    "rag_retrieval_chunks",
    "Returned chunks per search",
    buckets=(0, 1, 5, 10, 25, 50, 100),
    registry=None,
)
SCORES = Histogram(
    "rag_retrieval_similarity_score",
    "Returned cosine scores",
    buckets=(-1, -0.5, 0, 0.5, 0.75, 0.9, 1),
    registry=None,
)


def measure_upload[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    """Count service uploads without recording filenames, bytes, or IDs."""

    @wraps(function)
    def measured(*args: P.args, **kwargs: P.kwargs) -> T:
        outcome = "failure"
        try:
            result = function(*args, **kwargs)
            outcome = "success"
            return result
        finally:
            UPLOADS.labels(outcome).inc()

    return measured


def measure_retrieval[**P](
    function: Callable[P, RetrievalResponse],
) -> Callable[P, RetrievalResponse]:
    """Count every service search, including errors and empty results."""

    @wraps(function)
    def measured(*args: P.args, **kwargs: P.kwargs) -> RetrievalResponse:
        started = perf_counter()
        outcome = "failure"
        try:
            response = function(*args, **kwargs)
            outcome = "success"
            RESULTS.observe(len(response.results))
            for result in response.results:
                SCORES.observe(result.score)
            return response
        finally:
            RETRIEVALS.labels(outcome).inc()
            LATENCY.observe(perf_counter() - started)

    return measured


class WorkerCollector:
    """Aggregate durable measurements from independent Dagster processes via SQL."""

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def collect(self) -> Iterator[Metric]:
        observation = ProcessingObservation
        with self.factory() as session:
            rows = session.execute(
                select(
                    observation.outcome,
                    func.count(),
                    func.count(observation.duration_seconds),
                    func.sum(observation.duration_seconds),
                    func.sum(observation.chunk_count),
                    func.count(observation.embedding_duration_seconds),
                    func.sum(observation.embedding_duration_seconds),
                ).group_by(observation.outcome)
            ).all()
        runs = CounterMetricFamily(
            "rag_processing_runs", "Completed worker executions", labels=["outcome"]
        )
        durations = SummaryMetricFamily(
            "rag_processing_duration_seconds",
            "Completed worker duration",
            labels=["outcome"],
        )
        chunks = SummaryMetricFamily(
            "rag_processing_chunks",
            "Chunks published by successful workers",
            labels=["outcome"],
        )
        embeddings = SummaryMetricFamily(
            "rag_document_embedding_duration_seconds",
            "Document embedding batch duration",
            labels=["outcome"],
        )
        for (
            outcome,
            count,
            duration_count,
            duration,
            chunk_count,
            embedding_count,
            embedding_time,
        ) in rows:
            runs.add_metric([outcome], count)
            durations.add_metric([outcome], duration_count, duration or 0)
            chunks.add_metric([outcome], count, chunk_count)
            embeddings.add_metric([outcome], embedding_count, embedding_time or 0)
        yield from (runs, durations, chunks, embeddings)


def metrics_registry(factory: sessionmaker[Session] | None) -> CollectorRegistry:
    """Exclude default process/environment collectors and user-controlled labels."""
    registry = CollectorRegistry()
    for metric in (UPLOADS, RETRIEVALS, LATENCY, QUERY_EMBEDDING, RESULTS, SCORES):
        registry.register(metric)
    if factory is not None:
        registry.register(WorkerCollector(factory))
    return registry
