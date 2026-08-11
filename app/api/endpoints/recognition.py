import time
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, File, UploadFile
from pydantic import ValidationError

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.core.logging import logger
from app.schemas.plate import (
    PlateResult,
    ProviderEnum,
    ProvidersResponse,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.factory import PlateRecognizerFactory
from app.services.image_processing import decode_and_downscale
from app.services.yolo_filter import filter_vehicle_and_occupancy

router = APIRouter(tags=["ANPR Recognition"])

# Build the waterfall order so the operator-configured DEFAULT_PROVIDER is always
# tried first. The remaining providers follow in a fixed fallback sequence.
# Previously this was hard-coded, meaning DEFAULT_PROVIDER in .env was silently
# ignored by the waterfall even though /providers reported it as the default.
_FIXED_FALLBACK_ORDER = [
    ProviderEnum.PADDLEOCR,
    ProviderEnum.NVIDIA,
    ProviderEnum.PLATERECOGNIZER,
]
FALLBACK_PROVIDERS: list[ProviderEnum] = [
    settings.DEFAULT_PROVIDER,
    *[p for p in _FIXED_FALLBACK_ORDER if p != settings.DEFAULT_PROVIDER],
]

@router.get("/providers", response_model=ProvidersResponse, summary="List Supported Recognition Providers")
def list_providers():
    logger.debug("Listing available recognition providers.")
    return ProvidersResponse(
        available_providers=PlateRecognizerFactory.list_providers(),
        default_provider=settings.DEFAULT_PROVIDER
    )

def _validate_plate_results(
    raw_results: Any,
    provider: ProviderEnum,
) -> List[PlateResult]:
    """
    Turn provider output into PlateResult, validating each item (NASA rule 7).

    Previously `[PlateResult(**item) for item in raw_results]`. That trusts a
    strategy to return exactly the right keys: an unexpected key raises
    TypeError, a missing one raises ValidationError, and either becomes a 500
    on a request that had already succeeded at recognising a plate.

    One malformed entry should not discard the good ones, so entries are
    validated individually and bad ones are dropped with a log line rather than
    taking down the response.
    """
    if not isinstance(raw_results, list):
        logger.error(
            f"Provider '{provider.value}' returned {type(raw_results).__name__}, expected list."
        )
        return []

    validated: List[PlateResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            logger.warning(
                f"Provider '{provider.value}' returned a non-dict result "
                f"({type(item).__name__}); discarding."
            )
            continue
        try:
            validated.append(PlateResult.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                f"Provider '{provider.value}' returned an unusable result "
                f"{ {k: item.get(k) for k in list(item)[:4]} }: {exc.error_count()} error(s); discarding."
            )
    return validated


def _run_waterfall(
    image_bytes: bytes,
    filename: str,
    vehicle_box: Optional[tuple] = None,
) -> Tuple[List[PlateResult], ProviderEnum]:
    """
    Try each provider in order, returning the first usable result.

    Extracted from the handler per rule 4 — recognize_plate had grown past a
    page and was doing upload validation, pre-screening, provider orchestration
    and response assembly in one body.
    """
    for provider in FALLBACK_PROVIDERS:
        logger.info(f"Attempting recognition on '{filename}' using provider: '{provider.value}'")
        try:
            recognizer = PlateRecognizerFactory.get_recognizer(provider)
            raw_results = recognizer.recognize(
                image_bytes,
                filename=filename,
                vehicle_box=vehicle_box,
            )
        except ContractViolation:
            # An internal invariant broke. Failing over to the next provider
            # would just hide it behind a slower path, so let it surface.
            raise
        except Exception as exc:
            logger.error(
                f"Provider '{provider.value}' encountered an error: {exc}. "
                f"Falling back to next provider..."
            )
            continue

        plate_results = _validate_plate_results(raw_results, provider)
        if plate_results:
            logger.info(
                f"Successfully recognized {len(plate_results)} plate(s) in "
                f"'{filename}' via '{provider.value}'."
            )
            return plate_results, provider

        logger.warning(
            f"Provider '{provider.value}' returned no usable license plate results. "
            f"Falling back to next provider..."
        )

    return [], FALLBACK_PROVIDERS[0]


@router.post("/recognize", response_model=RecognitionResponse, summary="Extract License Plate from Image")
def recognize_plate(
    file: UploadFile = File(..., description="Vehicle image file (JPEG/PNG)")  # noqa: B008
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
    plate_results, active_provider = _run_waterfall(
        image_bytes,
        filename=file.filename or "image.jpg",
        vehicle_box=yolo_result.get("vehicle_box"),
    )

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
