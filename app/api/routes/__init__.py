"""API route submodules."""

from app.api.routes.info import router as info_router
from app.api.routes.recognition import router as recognition_router

__all__ = ["info_router", "recognition_router"]
