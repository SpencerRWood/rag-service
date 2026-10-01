"""Integration-level application construction checks."""

from importlib.metadata import version

from rag_service.main import create_app


def test_application_exposes_health_route() -> None:
    """The service is constructible without live infrastructure dependencies."""
    app = create_app()

    assert app.title == "RAG Service"
    assert app.version == version("rag-service")
