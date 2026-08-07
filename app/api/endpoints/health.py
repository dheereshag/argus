from fastapi import APIRouter

from app.core.config import settings
from app.schemas.plate import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse, summary="Check API Health Status")
def check_health():
    return HealthResponse(status="healthy", version=settings.VERSION)
