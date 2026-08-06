import time
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.schemas.plate import (
    RecognitionResponse,
    ProvidersResponse,
    PlateResult,
    ProviderEnum,
    RecognitionStatusEnum
)
from app.services.factory import PlateRecognizerFactory
from app.services.image_processing import decode_and_downscale
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
def recognize_plate(
    file: UploadFile = File(..., description="Vehicle image file (JPEG/PNG)")
):
    """
    Deliberately a sync `def`, not `async def`.

    Every unit of work below blocks: YOLO inference, PaddleOCR, and the provider
    HTTP calls are all synchronous CPU or socket work. Declaring the handler
    `async` ran that work directly on the event loop, so the process served one
    request at a time and /health stopped answering while a recognition was in
    flight — which is exactly when an orchestrator restarts the pod.

    FastAPI runs sync handlers in an anyio worker threadpool, so this keeps the
    loop free. Do not add `async` back without first moving the blocking work
    into a threadpool or process pool.
    """
    logger.info(f"Received recognition request for file '{file.filename}'.")

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Rejected invalid file type '{file.content_type}' for file '{file.filename}'.")
        raise InvalidImageError(f"File '{file.filename}' is not a valid image format.")

    start_time = time.time()

    # Read at most one byte past the limit so an oversized body is detected
    # without ever holding the whole thing in memory.
    raw_bytes = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > settings.MAX_UPLOAD_BYTES:
        logger.warning(
            f"Rejected oversized upload for '{file.filename}': "
            f"exceeds {settings.MAX_UPLOAD_BYTES} bytes."
        )
        raise PayloadTooLargeError(
            f"Upload exceeds the maximum permitted size of "
            f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    if not raw_bytes:
        raise InvalidImageError(f"File '{file.filename}' is empty.")

    # Guards against decompression bombs and normalises to a bounded JPEG.
    # Every downstream stage (YOLO, ROI crops, provider uploads) sees these
    # bytes, so box coordinates stay consistent with the image they came from.
    image_bytes = decode_and_downscale(raw_bytes)

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
            raw_results = recognizer.recognize(
                image_bytes,
                filename=file.filename,
                vehicle_box=yolo_result.get("vehicle_box")
            )
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
