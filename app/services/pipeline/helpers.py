"""Input resolution, coordinate translation, and output validation helpers."""

from typing import Any

from pydantic import ValidationError

from app.core.exceptions import InvalidImageError
from app.schemas import PlateResult


def validate_plate_results(
    raw_results: Any,
    vehicle_type: str | None = None,
) -> list[PlateResult]:
    """Validate and convert raw dictionary outputs into PlateResult Pydantic schemas."""
    if not isinstance(raw_results, list):
        return []
    validated: list[PlateResult] = []
    for item in raw_results:
        if isinstance(item, dict):
            entry = dict(item)
            if vehicle_type is not None and "vehicle_type" not in entry:
                entry["vehicle_type"] = vehicle_type
            try:
                validated.append(PlateResult.model_validate(entry))
            except ValidationError:
                continue
    return validated


def _resolve_bytes(image_input: str | bytes) -> bytes:
    """Read polymorphic image input into raw in-memory bytes."""
    if isinstance(image_input, bytes):
        return image_input
    if isinstance(image_input, str):
        try:
            with open(image_input, "rb") as f:
                return f.read()
        except Exception as e:
            raise InvalidImageError(f"Failed to read image file '{image_input}': {e}") from e
    raise InvalidImageError(f"Unsupported image input type: {type(image_input).__name__}")


def _adjust_crop_coordinates(
    raw_results: list[dict[str, Any]],
    crop_box: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    """Translate plate bounding box coordinates from crop space to full image frame space."""
    if not crop_box:
        return raw_results
    cx1, cy1, _, _ = crop_box
    adjusted: list[dict[str, Any]] = []
    for item in raw_results:
        entry = dict(item)
        box = entry.get("box")
        if box and isinstance(box, (tuple, list)) and len(box) >= 4:
            entry["box"] = (int(box[0] + cx1), int(box[1] + cy1), int(box[2] + cx1), int(box[3] + cy1))
        adjusted.append(entry)
    return adjusted
