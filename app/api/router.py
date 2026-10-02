"""Aggregated API router assembling all route submodules."""

from fastapi import APIRouter

from app.api.routes.info import router as info_router
from app.api.routes.recognition import router as recognition_router

api_router = APIRouter()
api_router.include_router(info_router)
api_router.include_router(recognition_router)
