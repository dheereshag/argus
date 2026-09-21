"""Stage 2 OCR execution and per-vehicle cropping."""

from typing import Any

from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas import DetectedVehicle, DetectionResult, PlateResult
from app.services.image_processing import ImageInput
from app.services.pipeline.helpers import (
    _adjust_crop_coordinates,
    validate_plate_results,
)


def _ocr_single_vehicle(
    recognizer: Any,
    vehicle: DetectedVehicle,
    image_input: ImageInput,
    filename: str,
    allow_fallback: bool = False,
) -> list[PlateResult]:
    """Execute RapidOCR on a single vehicle crop with coordinate translation and fallback."""
    target = vehicle.crop if vehicle.crop is not None else image_input
    raw = recognizer.recognize(target, filename=filename)

    if vehicle.crop is not None:
        if any(r.get("plate") and r.get("plate") != "N/A" for r in raw):
            raw = _adjust_crop_coordinates(raw, vehicle.crop_box)
        elif allow_fallback:
            raw = recognizer.recognize(image_input, filename=filename)

    return validate_plate_results(raw, vehicle_type=vehicle.vehicle_type)


def _run_stage2_ocr(detection: DetectionResult, image_input: ImageInput, filename: str) -> list[PlateResult]:
    """Execute RapidOCR across detected vehicles, falling back to full frame if needed."""
    logger.info(f"Running OCR on '{filename}'")
    try:
        import app.services.pipeline as pl

        recognizer = pl.PlateRecognizer()
        if not detection.vehicles:
            raw = recognizer.recognize(image_input, filename=filename)
            return validate_plate_results(raw, vehicle_type=None)

        results: list[PlateResult] = []
        allow_fallback = len(detection.vehicles) == 1
        for vehicle in detection.vehicles:
            results.extend(_ocr_single_vehicle(recognizer, vehicle, image_input, filename, allow_fallback))

        valid = [r for r in results if r.plate != "N/A"]
        dedup: list[PlateResult] = []
        for p in valid:
            if not any(p.plate == e.plate and p.box and e.box and max(abs(p.box[0] - e.box[0]), abs(p.box[1] - e.box[1])) < 30 for e in dedup):
                dedup.append(p)
        return dedup if dedup else results
    except (ANPRServiceError, ValueError, RuntimeError, OSError, KeyError, AttributeError) as exc:
        logger.error(f"OCR failed on '{filename}': {exc}")
        return []
