import time
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query
from app.core.config import settings
from app.core.exceptions import InvalidImageError
from app.schemas.plate import (
    RecognitionResponse,
    ProvidersResponse,
    PlateResult,
    ProviderEnum,
    RecognitionStatusEnum
)
from app.services.factory import PlateRecognizerFactory
from app.services.yolo_filter import filter_vehicle_and_occupancy

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

    # Step 1: YOLO Pre-screening (Check 4-wheeler vehicle present AND no human present)
    yolo_result = filter_vehicle_and_occupancy(image_bytes)

    if not yolo_result["is_eligible"]:
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        active_provider = provider or settings.DEFAULT_PROVIDER
        return RecognitionResponse(
            success=False,
            status=yolo_result["status"],
            status_message=yolo_result["status_message"],
            vehicle_detected=yolo_result["vehicle_detected"],
            human_detected=yolo_result["human_detected"],
            filename=file.filename,
            provider=active_provider,
            results=[],
            execution_time_ms=execution_time_ms
        )

    # Step 2: License Plate Recognition
    recognizer = PlateRecognizerFactory.get_recognizer(provider)
    active_provider = provider or settings.DEFAULT_PROVIDER

    raw_results = recognizer.recognize(image_bytes, filename=file.filename)
    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    plate_results = [PlateResult(**item) for item in raw_results]

    if plate_results:
        final_status = RecognitionStatusEnum.SUCCESS
        status_msg = "License plate successfully detected and recognized."
        success_flag = True
    else:
        final_status = RecognitionStatusEnum.NO_PLATE_DETECTED
        status_msg = "4-wheeler detected with no human, but no license plate could be extracted."
        success_flag = False

    return RecognitionResponse(
        success=success_flag,
        status=final_status,
        status_message=status_msg,
        vehicle_detected=yolo_result["vehicle_detected"],
        human_detected=yolo_result["human_detected"],
        filename=file.filename,
        provider=active_provider,
        results=plate_results,
        execution_time_ms=execution_time_ms
    )
