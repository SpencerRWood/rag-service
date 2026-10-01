"""Unit tests for the liveness endpoint."""

from rag_service.api.routes.health import read_health


def test_read_health_returns_service_status() -> None:
    """The liveness check must not require database or worker connectivity."""
    assert read_health() == {"service": "rag-service", "status": "ok"}
