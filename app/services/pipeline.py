import time
from typing import Any

from pydantic import ValidationError

from app.core.exceptions import (
    ANPRServiceError,
    InvalidImageError,
)
from app.core.logging import logger
from app.schemas import (
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.image_processing import decode_and_downscale, load_rgb
from app.services.ocr import PlateRecognizer
from app.services.yolo_filter import filter_vehicle_and_occupancy


def validate_plate_results(raw_results: Any) -> list[PlateResult]:
    if not isinstance(raw_results, list):
        logger.error(f"PlateRecognizer returned {type(raw_results).__name__}, expected list.")
        return []

    validated: list[PlateResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            logger.warning(f"PlateRecognizer returned non-dict result ({type(item).__name__}); discarding.")
            continue
        try:
            validated.append(PlateResult.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                f"PlateRecognizer returned unusable result "
                f"{ {k: item.get(k) for k in list(item)[:4]} }: {exc.error_count()} error(s); discarding."
            )
    return validated


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """
    Main ANPR entry point. Performs YOLO v11 pre-screening, vehicle cropping, then RapidOCR plate extraction.
    """
    start_time = time.time()

    if isinstance(image_input, str):
        filename = filename or image_input
        try:
            with open(image_input, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            raise InvalidImageError(f"Failed to read image file '{image_input}': {e}") from e
    elif isinstance(image_input, bytes):
        raw_bytes = image_input
    else:
        raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")

    image_bytes = decode_and_downscale(raw_bytes)

    # Step 1: YOLO Pre-screening
    yolo_result = filter_vehicle_and_occupancy(image_bytes)

    if not yolo_result["is_eligible"]:
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Image '{filename}' ineligible: {yolo_result['status_message']} ({execution_time_ms} ms)")
        return RecognitionResponse(
            success=False,
            rejected=True,
            status=yolo_result["status"],
            status_message=yolo_result["status_message"],
            vehicle_detected=yolo_result["vehicle_detected"],
            vehicle_type=yolo_result["vehicle_type"],
            human_detected=yolo_result["human_detected"],
            filename=filename,
            results=[],
            execution_time_ms=execution_time_ms,
        )

    # Step 2: Vehicle Cropping & OCR
    logger.info(f"Running OCR on '{filename}'")
    try:
        recognizer = PlateRecognizer()
        if yolo_result["vehicle_box"] is not None:
            pil_img = load_rgb(image_bytes)
            vehicle_crop = pil_img.crop(yolo_result["vehicle_box"])
            raw_results = recognizer.recognize(vehicle_crop, filename=filename)
            # Fall back to full frame if vehicle crop yielded no plate match
            if not any(r.get("plate") and r.get("plate") != "N/A" for r in raw_results):
                raw_results = recognizer.recognize(image_bytes, filename=filename)
        else:
            raw_results = recognizer.recognize(image_bytes, filename=filename)
    except (ANPRServiceError, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
        logger.error(f"OCR failed on '{filename}': {exc}")
        raw_results = []

    plate_results = validate_plate_results(raw_results)

    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    has_valid_plate = any(r.plate != "N/A" for r in plate_results)
    final_status = RecognitionStatusEnum.SUCCESS if has_valid_plate else RecognitionStatusEnum.NO_PLATE_DETECTED

    if has_valid_plate:
        target_name = yolo_result["vehicle_type"] or ("vehicle" if yolo_result["vehicle_detected"] else None)
        if target_name:
            status_msg = f"License plate successfully detected and recognized on {target_name}."
        else:
            status_msg = "License plate successfully detected and recognized."
    else:
        if yolo_result["vehicle_detected"]:
            status_msg = (
                f"4-wheeler ({yolo_result['vehicle_type'] or 'vehicle'}) detected, "
                f"but no readable license plate characters could be recognized."
            )
        else:
            status_msg = "No vehicle detected and no readable license plate characters could be recognized."

    return RecognitionResponse(
        success=has_valid_plate,
        rejected=False,
        status=final_status,
        status_message=status_msg,
        vehicle_detected=yolo_result["vehicle_detected"],
        vehicle_type=yolo_result["vehicle_type"],
        human_detected=yolo_result["human_detected"],
        filename=filename,
        results=plate_results,
        execution_time_ms=execution_time_ms,
    )
