from fastapi import APIRouter

from app.core.config import settings
from app.schemas.plate import ProvidersResponse
from app.services.factory import PlateRecognizerFactory

router = APIRouter()


@router.get(
    "/providers",
    response_model=ProvidersResponse,
    summary="List Recognition Providers",
    description="Retrieve the list of supported ANPR recognition engine providers and the current default provider.",
    tags=["Providers"],
)
async def list_providers() -> ProvidersResponse:
    providers = PlateRecognizerFactory.list_providers()
    return ProvidersResponse(
        available_providers=providers,
        default_provider=settings.DEFAULT_PROVIDER,
    )
