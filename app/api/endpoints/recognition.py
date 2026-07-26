import time
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query
from app.core.config import settings
from app.core.logging import logger
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

FALLBACK_PROVIDERS = [
    ProviderEnum.PADDLEOCR,
    ProviderEnum.NVIDIA,
    ProviderEnum.PLATERECOGNIZER,
]

@router.get("/providers", response_model=ProvidersResponse, summary="List Supported Recognition Providers")
def list_providers():
    logger.debug("Listing available recognition providers.")
    return ProvidersResponse(
        available_providers=PlateRecognizerFactory.list_providers(),
        default_provider=settings.DEFAULT_PROVIDER
    )

@router.post("/recognize", response_model=RecognitionResponse, summary="Extract License Plate from Image")
async def recognize_plate(
    file: UploadFile = File(..., description="Vehicle image file (JPEG/PNG)")
):
    logger.info(f"Received recognition request for file '{file.filename}'.")

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Rejected invalid file type '{file.content_type}' for file '{file.filename}'.")
        raise InvalidImageError(f"File '{file.filename}' is not a valid image format.")

    start_time = time.time()
    image_bytes = await file.read()

    # Step 1: YOLO Pre-screening (Check 4-wheeler vehicle present AND no human present)
    yolo_result = filter_vehicle_and_occupancy(image_bytes)

    if not yolo_result["is_eligible"]:
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Image '{file.filename}' ineligible: {yolo_result['status_message']} ({execution_time_ms} ms)")
        return RecognitionResponse(
            success=False,
            status=yolo_result["status"],
            status_message=yolo_result["status_message"],
            vehicle_detected=yolo_result["vehicle_detected"],
            vehicle_type=yolo_result["vehicle_type"],
            human_detected=yolo_result["human_detected"],
            filename=file.filename,
            provider=FALLBACK_PROVIDERS[0],
            results=[],
            execution_time_ms=execution_time_ms
        )

    # Step 2: License Plate Recognition with Waterfall Fallback Mechanism
    plate_results = []
    active_provider = FALLBACK_PROVIDERS[0]

    for provider in FALLBACK_PROVIDERS:
        logger.info(f"Attempting recognition on '{file.filename}' using provider: '{provider.value}'")
        try:
            recognizer = PlateRecognizerFactory.get_recognizer(provider)
            raw_results = recognizer.recognize(image_bytes, filename=file.filename)
            if raw_results:
                plate_results = [PlateResult(**item) for item in raw_results]
                active_provider = provider
                logger.info(f"Successfully recognized {len(plate_results)} plate(s) in '{file.filename}' via '{provider.value}'.")
                break
            else:
                logger.warning(f"Provider '{provider.value}' returned no license plate results. Falling back to next provider...")
        except Exception as e:
            logger.error(f"Provider '{provider.value}' encountered an error: {e}. Falling back to next provider...")

    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    if plate_results:
        final_status = RecognitionStatusEnum.SUCCESS
        status_msg = f"License plate successfully detected and recognized on {yolo_result['vehicle_type']} via {active_provider.value}."
        success_flag = True
    else:
        final_status = RecognitionStatusEnum.NO_PLATE_DETECTED
        status_msg = f"4-wheeler ({yolo_result['vehicle_type']}) detected with no human, but no license plate could be extracted across fallback providers."
        success_flag = False
        active_provider = FALLBACK_PROVIDERS[0]
        logger.info(f"No license plate detected in '{file.filename}' across all fallback providers ({execution_time_ms} ms).")

    return RecognitionResponse(
        success=success_flag,
        status=final_status,
        status_message=status_msg,
        vehicle_detected=yolo_result["vehicle_detected"],
        vehicle_type=yolo_result["vehicle_type"],
        human_detected=yolo_result["human_detected"],
        filename=file.filename,
        provider=active_provider,
        results=plate_results,
        execution_time_ms=execution_time_ms
    )
