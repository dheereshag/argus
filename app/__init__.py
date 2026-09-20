"""
Argus: Production Indian Automatic Number Plate Recognition (ANPR) Python Library & CLI.
"""

from app.schemas import (
    DetectedVehicle,
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.pipeline import recognize_plate_image

__all__ = [
    "DetectedVehicle",
    "PlateResult",
    "RecognitionResponse",
    "RecognitionStatusEnum",
    "recognize_plate_image",
]
