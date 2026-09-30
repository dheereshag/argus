"""ANPR Pipeline Orchestrator: 4-tier cascaded fallback with thin response."""

import time
from typing import Any

from app.core.logging import logger
from app.schemas import DetectionResult, RecognitionResponse
from app.services.image_processing import decode_image
from app.services.ocr.fast_alpr_runner import (
    run_fast_alpr_on_crops,
    run_fast_alpr_pipeline,
)
from app.services.pipeline.helpers import _resolve_bytes
from app.services.pipeline.response import _build_response
from app.services.pipeline.stages import _run_stage2_ocr


def _run_rapidocr_fallback(det: DetectionResult, img: Any, fn: str) -> list[str]:
    """Tier 3 & 4: Argus RapidOCR on vehicle crops with full-frame fallback."""
    logger.info(f"Fast-ALPR produced no plates. Falling back to RapidOCR for '{fn}'")
    fallback_results = _run_stage2_ocr(det, img, fn)
    return [r.plate for r in fallback_results if r.plate and r.plate != "N/A"]


def _execute_cascade(prepared_img: Any, filename: str) -> list[str]:
    """Execute 4-tier cascade: Fast-ALPR full -> Fast-ALPR crop -> RapidOCR crop -> RapidOCR full."""
    # Tier 1: Fast-ALPR directly on full frame
    if plates := run_fast_alpr_pipeline(prepared_img):
        return plates

    # Tier 2: Vehicle detection + Fast-ALPR on vehicle crops
    import app.services.pipeline as pl

    detection = pl.VehicleDetector().detect(prepared_img)
    if plates := run_fast_alpr_on_crops(detection.vehicles):
        return plates

    # Tier 3 & 4: RapidOCR on vehicle crops, falling back to full-frame RapidOCR
    return _run_rapidocr_fallback(detection, prepared_img, filename)


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """Process image through the 4-tier cascaded ANPR pipeline."""
    start_time = time.time()
    resolved_fn = filename or (image_input if isinstance(image_input, str) else "image.jpg")
    prepared_img = decode_image(_resolve_bytes(image_input))
    plates = _execute_cascade(prepared_img, resolved_fn)
    return _build_response(plates, start_time)
