import time
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query
from app.core.config import settings
from app.core.exceptions import InvalidImageError
from app.schemas.plate import RecognitionResponse, ProvidersResponse, PlateResult, ProviderEnum
from app.services.factory import PlateRecognizerFactory

router = APIRouter(tags=["ANPR Recognition"])

@router.get("/providers", response_model=ProvidersResponse, summary="List Supported Recognition Providers")
def list_providers():
    return ProvidersResponse(
        available_providers=PlateRecognizerFactory.list_providers(),
        default_provider=settings.DEFAULT_PROVIDER
    )

@router.post("/recognize", response_model=RecognitionResponse, summary="Extract License Plate from Image")
async def recognize_plate(
    file: UploadFile = File(..., description="Vehicle image file (JPEG/PNG)"),
    provider: Optional[ProviderEnum] = Query(
        None,
        description="Recognition provider choice: 'platerecognizer' or 'nvidia'. Defaults to .env setting."
    )
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise InvalidImageError(f"File '{file.filename}' is not a valid image format.")

    start_time = time.time()
    image_bytes = await file.read()

    # Obtain model strategy from Factory
    recognizer = PlateRecognizerFactory.get_recognizer(provider)
    active_provider = provider or settings.DEFAULT_PROVIDER

    # Run plate recognition
    raw_results = recognizer.recognize(image_bytes, filename=file.filename)
    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    plate_results = [PlateResult(**item) for item in raw_results]

    return RecognitionResponse(
        success=True,
        filename=file.filename,
        provider=active_provider,
        results=plate_results,
        execution_time_ms=execution_time_ms
    )
