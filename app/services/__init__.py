"""
Argus ANPR core computer vision and inference services.

Exports:
  - VehicleDetector: Stage 1 YOLO v11 vehicle detection and occupancy gatekeeper.
  - PlateRecognizer: Stage 2 RapidOCR engine with 2D spatial candidate pairing.
"""

from app.services.detector import VehicleDetector
from app.services.ocr import PlateRecognizer

__all__ = [
    "PlateRecognizer",
    "VehicleDetector",
]

