"""Zero-vehicle detection fallback logic."""

from app.core.config import settings
from app.core.logging import logger
from app.schemas import DetectionResult, PlateResult, RecognitionStatusEnum
from app.services.pipeline.stages import _run_stage2_ocr


def _attempt_zero_vehicle_fallback(
    detection: DetectionResult,
    image_bytes: bytes,
    resolved_filename: str,
) -> list[PlateResult] | None:
    """Attempt full-frame OCR when YOLO misses a vehicle (e.g. half-in-frame / tight crop)."""
    if (
        detection.status == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
        and settings.FALLBACK_OCR_ON_NO_VEHICLE
    ):
        logger.info(
            f"No 4-wheeler detected on '{resolved_filename}', attempting fallback OCR..."
        )
        plate_results = _run_stage2_ocr(detection, image_bytes, resolved_filename)
        if any(r.plate != "N/A" for r in plate_results):
            logger.info(f"Fallback OCR succeeded for '{resolved_filename}' without vehicle bbox")
            return plate_results
    return None
