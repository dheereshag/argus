"""
Argus: Production Indian Automatic Number Plate Recognition (ANPR) Python Library & Service.
"""

from app.schemas import (
    DetectedVehicle,
    PlateResult,
    RecognitionResponse,
)
from app.services.pipeline import recognize_plate_image

__all__ = [
    "DetectedVehicle",
    "PlateResult",
    "RecognitionResponse",
    "recognize_plate_image",
]
