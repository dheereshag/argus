"""
Argus: Production Indian Automatic Number Plate Recognition (ANPR) Python Library & CLI.
"""

from app.schemas.plate import (
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)
from app.services.pipeline import recognize_plate_image

__all__ = [
    "PlateResult",
    "RecognitionResponse",
    "RecognitionStatusEnum",
    "recognize_plate_image",
]
