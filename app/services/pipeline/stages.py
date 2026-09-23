"""Stage 2 OCR execution: per-vehicle crop pass + full-frame spatial association pass."""

from typing import Any

from app.core.config import settings
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas import DetectedVehicle, DetectionResult, PlateResult
from app.services.image_processing import ImageInput
from app.services.pipeline.association import associate_fullframe_plates
from app.services.pipeline.fallback import run_full_frame_fallback
from app.services.pipeline.helpers import (
    _adjust_crop_coordinates,
    validate_plate_results,
)


def _ocr_single_vehicle(
    recognizer: Any,
    vehicle: DetectedVehicle,
    filename: str,
) -> list[PlateResult]:
    """Execute RapidOCR on a single vehicle crop with coordinate translation back to frame space."""
    if vehicle.crop is None:
        return []
    raw = recognizer.recognize(vehicle.crop, filename=filename)
    if any(r.get("plate") and r.get("plate") != "N/A" for r in raw):
        raw = _adjust_crop_coordinates(raw, vehicle.crop_box)
    return validate_plate_results(raw, vehicle_type=vehicle.vehicle_type)


def _run_stage2_ocr(detection: DetectionResult, image_input: ImageInput, filename: str) -> list[PlateResult]:
    """Execute per-vehicle crop OCR then full-frame spatial association to capture all plates."""
    logger.info(f"Running OCR on '{filename}'")
    try:
        import app.services.pipeline as pl

        recognizer = pl.PlateRecognizer()
        if not detection.vehicles:
            return run_full_frame_fallback(recognizer, image_input, filename)

        crop_results: list[PlateResult] = []
        plated: set[int] = set()
        for vehicle in detection.vehicles:
            plates = _ocr_single_vehicle(recognizer, vehicle, filename)
            if plates:
                crop_results.extend(plates)
                plated.add(id(vehicle))

        run_ff = settings.ENABLE_FULL_FRAME_OCR or not crop_results
        raw_ff = recognizer.recognize(image_input, filename=filename) if run_ff else []
        return associate_fullframe_plates(raw_ff, detection.vehicles, crop_results, plated)
    except (ANPRServiceError, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
        logger.error(f"OCR failed on '{filename}': {exc}")
        return []
