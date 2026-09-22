"""Zero-vehicle full-frame OCR fallback routine."""

from typing import Any

from app.core.logging import logger
from app.schemas import PlateResult
from app.services.image_processing import ImageInput
from app.services.pipeline.helpers import validate_plate_results


def run_full_frame_fallback(
    recognizer: Any,
    image_input: ImageInput,
    filename: str,
) -> list[PlateResult]:
    """Execute full-frame OCR when no vehicle entities were localized."""
    logger.info(f"Running full-frame OCR fallback on '{filename}'")
    raw = recognizer.recognize(image_input, filename=filename)
    return validate_plate_results(raw, vehicle_type=None)

