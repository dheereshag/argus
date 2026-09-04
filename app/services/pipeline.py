import time
from typing import Any

from pydantic import ValidationError

from app.core.exceptions import ANPRServiceError, InvalidImageError
from app.core.logging import logger
from app.schemas import (
    DetectionResult,
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.detector import filter_vehicle_and_occupancy
from app.services.image_processing import decode_and_downscale
from app.services.ocr import PlateRecognizer


def validate_plate_results(raw_results: Any) -> list[PlateResult]:
    """Validate and filter raw dictionary outputs into PlateResult schemas."""
    if not isinstance(raw_results, list):
        return []
    validated: list[PlateResult] = []
    for item in raw_results:
        if isinstance(item, dict):
            try:
                validated.append(PlateResult.model_validate(item))
            except ValidationError:
                continue
    return validated


def _resolve_bytes(image_input: str | bytes) -> bytes:
    if isinstance(image_input, bytes):
        return image_input
    if isinstance(image_input, str):
        try:
            with open(image_input, "rb") as f:
                return f.read()
        except Exception as e:
            raise InvalidImageError(f"Failed to read image file '{image_input}': {e}") from e
    raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")


def _build_response(
    detection: DetectionResult,
    filename: str,
    start_time: float,
    *,
    success: bool,
    rejected: bool,
    status: RecognitionStatusEnum,
    message: str,
    results: list[PlateResult] | None = None,
) -> RecognitionResponse:
    return RecognitionResponse(
        success=success,
        rejected=rejected,
        status=status,
        status_message=message,
        vehicle_detected=detection.vehicle_detected,
        vehicle_type=detection.vehicle_type,
        human_detected=detection.human_detected,
        filename=filename,
        results=results or [],
        execution_time_ms=round((time.time() - start_time) * 1000, 2),
    )


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """
    Two-stage ANPR Pipeline:
    - Stage 1: VehicleDetector (YOLO localization, occupancy policies, vehicle crop)
    - Stage 2: PlateRecognizer (RapidOCR text recognition & Indian plate regex parsing)
    """
    start_time = time.time()
    resolved_filename = filename or (image_input if isinstance(image_input, str) else "image.jpg")
    image_bytes = decode_and_downscale(_resolve_bytes(image_input))

    # Stage 1: Vehicle Detection & Occupancy Gatekeeping
    detection = filter_vehicle_and_occupancy(image_bytes)
    if not detection.is_eligible:
        logger.info(f"Image '{resolved_filename}' ineligible: {detection.status_message}")
        return _build_response(
            detection,
            resolved_filename,
            start_time,
            success=False,
            rejected=True,
            status=detection.status or RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
            message=detection.status_message,
        )

    # Stage 2: License Plate Recognition
    logger.info(f"Running OCR on '{resolved_filename}'")
    raw: list[dict[str, Any]] = []
    try:
        recognizer = PlateRecognizer()
        target = detection.crop if detection.crop is not None else image_bytes
        raw = recognizer.recognize(target, filename=resolved_filename)
        # Fall back to full frame if vehicle crop yielded no plate candidate
        if detection.crop is not None and not any(r.get("plate") and r.get("plate") != "N/A" for r in raw):
            raw = recognizer.recognize(image_bytes, filename=resolved_filename)
    except (ANPRServiceError, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
        logger.error(f"OCR failed on '{resolved_filename}': {exc}")

    plate_results = validate_plate_results(raw)
    has_plate = any(r.plate != "N/A" for r in plate_results)
    final_status = RecognitionStatusEnum.SUCCESS if has_plate else RecognitionStatusEnum.NO_PLATE_DETECTED

    if has_plate:
        target_name = detection.vehicle_type or ("vehicle" if detection.vehicle_detected else None)
        msg = f"License plate successfully detected and recognized on {target_name}." if target_name else "License plate successfully detected and recognized."
    else:
        msg = (
            f"4-wheeler ({detection.vehicle_type or 'vehicle'}) detected, but no readable license plate characters could be recognized."
            if detection.vehicle_detected
            else "No vehicle detected and no readable license plate characters could be recognized."
        )

    return _build_response(
        detection,
        resolved_filename,
        start_time,
        success=has_plate,
        rejected=False,
        status=final_status,
        message=msg,
        results=plate_results,
    )
