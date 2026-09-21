"""End-to-End Two-Stage ANPR Pipeline Orchestrator."""

import time

from app.core.logging import logger
from app.schemas import DetectionResult, RecognitionResponse, RecognitionStatusEnum
from app.services.image_processing import ImageInput, decode_and_downscale
from app.services.pipeline.fallback import _attempt_zero_vehicle_fallback
from app.services.pipeline.helpers import _resolve_bytes
from app.services.pipeline.response import _build_response
from app.services.pipeline.stages import _run_stage2_ocr


def _handle_ineligible(
    detection: DetectionResult, image_input: ImageInput, filename: str, start_time: float
) -> RecognitionResponse:
    fallback = _attempt_zero_vehicle_fallback(detection, image_input, filename)
    if fallback is not None:
        return _build_response(
            detection, filename, start_time, success=True, rejected=False,
            status=RecognitionStatusEnum.SUCCESS, results=fallback,
        )
    logger.info(f"Image '{filename}' ineligible: status={detection.status}")
    st = detection.status or RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    return _build_response(detection, filename, start_time, success=False, rejected=True, status=st)


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """Execute the Two-Stage ANPR Pipeline (YOLO11 detection + RapidOCR) on an input image."""
    start_time = time.time()
    resolved_filename = filename or (image_input if isinstance(image_input, str) else "image.jpg")
    prepared_img = decode_and_downscale(_resolve_bytes(image_input))

    import app.services.pipeline as pl

    detection = pl.VehicleDetector().detect(prepared_img)
    if not detection.is_eligible:
        return _handle_ineligible(detection, prepared_img, resolved_filename, start_time)

    plate_results = _run_stage2_ocr(detection, prepared_img, resolved_filename)
    has_plate = any(r.plate != "N/A" for r in plate_results)
    st = RecognitionStatusEnum.SUCCESS if has_plate else RecognitionStatusEnum.NO_PLATE_DETECTED
    return _build_response(
        detection, resolved_filename, start_time,
        success=has_plate, rejected=False, status=st, results=plate_results,
    )
