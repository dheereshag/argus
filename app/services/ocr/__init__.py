"""Stage 2: License Plate Text Recognition (OCR) and Spatial Candidate Selection."""

from app.schemas import OCRToken
from app.services.ocr.recognizer import PlateRecognizer
from app.services.ocr.suppression import suppress_overlapping_candidates
from app.services.plate_rules import parse_plate_info

__all__ = [
    "OCRToken",
    "PlateRecognizer",
    "parse_plate_info",
    "suppress_overlapping_candidates",
]

