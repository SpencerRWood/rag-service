"""Top-level API router."""

from fastapi import APIRouter

from rag_service.api.routes.documents import router as document_router
from rag_service.api.routes.health import router as health_router
from rag_service.api.routes.knowledge_bases import router as knowledge_base_router
from rag_service.api.routes.metadata import router as metadata_router
from rag_service.api.routes.processing import router as processing_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(knowledge_base_router)
api_router.include_router(document_router)
api_router.include_router(processing_router)
api_router.include_router(metadata_router)
