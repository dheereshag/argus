from fastapi import APIRouter

from app.api.endpoints import recognition

api_router = APIRouter()

api_router.include_router(recognition.router)
