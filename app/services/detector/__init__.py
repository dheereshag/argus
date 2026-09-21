"""Stage 1: YOLO26 Vehicle Detection, Occupancy Policy Gatekeeper, and Cropper."""

from ultralytics import YOLO

from app.core.config import settings
from app.services.detector.detector import VehicleDetector
from app.services.detector.geometry import (
    BoundingBox,
    box_iou,
    clamp_box,
    is_contained,
    pad_box,
)
from app.services.detector.occupancy import evaluate_occupancy

__all__ = [
    "YOLO",
    "BoundingBox",
    "VehicleDetector",
    "box_iou",
    "clamp_box",
    "evaluate_occupancy",
    "is_contained",
    "pad_box",
    "settings",
]


