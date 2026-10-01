"""Typed resource injection for future ingestion and reprocessing jobs."""

from dagster import ConfigurableResource

from rag_service.config import Settings, load_settings
from rag_service.storage import FileStorage, build_storage


class RAGResource(ConfigurableResource["RAGResource"]):
    """Resolve settings and secrets only when application work needs them."""

    def settings(self) -> Settings:
        """Share the API's runtime configuration contract."""
        return load_settings()

    def storage(self) -> FileStorage:
        """Build the same portable adapter as application services."""
        return build_storage(self.settings())
