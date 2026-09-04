from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.constants import (
    FOUR_WHEELER_CLASS_NAMES,
    MAX_DETECTIONS,
    MIN_CROP_EDGE_PX,
    PERSON_CLASS_ID,
)
from app.core.config import settings
from app.core.contracts import bounded, ensure, require
from app.core.logging import logger
from app.schemas import DetectionResult, RecognitionStatusEnum
from app.services.image_processing import ImageInput, load_rgb

type BoundingBox = tuple[int, int, int, int]

_YOLO_MODEL: YOLO | None = None


def get_yolo_model() -> YOLO:
    """Return the singleton YOLO v11 model instance, loading weights on first call."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        target_model = settings.YOLO_MODEL_NAME if "11" in (settings.YOLO_MODEL_NAME or "") else "yolo11n.pt"
        logger.debug(f"Loading YOLO model weights: {target_model}")
        _YOLO_MODEL = YOLO(target_model)
    ensure(_YOLO_MODEL is not None, "YOLO model failed to initialise")
    return _YOLO_MODEL


def clamp_box(box: Any, width: int, height: int) -> BoundingBox | None:
    """Clamp an xyxy box to real image bounds. Returns None when malformed or degenerate."""
    require(width > 0 and height > 0, f"image dimensions must be positive, got {width}x{height}")
    if not box or not isinstance(box, (tuple, list)) or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))

    if x2 - x1 < MIN_CROP_EDGE_PX or y2 - y1 < MIN_CROP_EDGE_PX:
        return None
    return (x1, y1, x2, y2)


class VehicleDetector:
    """Stage 1: YOLO v11 Vehicle Detection, Occupancy Policy Gatekeeper, and Cropper."""

    def __init__(self) -> None:
        self.get_model = get_yolo_model

    def _run_detection(
        self,
        pil_img: Image.Image,
        human_conf_thresh: float,
        vehicle_conf_thresh: float,
    ) -> tuple[bool, list[tuple[str, BoundingBox]]]:
        """Run YOLO and extract (human_present, area-sorted vehicles)."""
        require(pil_img is not None, "_run_detection called with no image")
        width, height = pil_img.size
        results = next(iter(self.get_model()(pil_img, verbose=False)))

        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cls"):
            return False, []

        def to_np(arr: Any) -> np.ndarray:
            return arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)

        cls_ids = to_np(boxes.cls)
        confs = to_np(boxes.conf)
        xyxy = to_np(boxes.xyxy) if getattr(boxes, "xyxy", None) is not None else None

        human_detected = False
        vehicles: list[tuple[int, str, BoundingBox]] = []

        for idx, (raw_cls, conf) in enumerate(
            bounded(list(zip(cls_ids, confs, strict=False)), MAX_DETECTIONS, "YOLO detections")
        ):
            cls_id = int(raw_cls)
            if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh:
                human_detected = True
                continue

            if cls_id not in FOUR_WHEELER_CLASS_NAMES or conf < vehicle_conf_thresh:
                continue

            raw_box: tuple[int, int, int, int] | None = None
            if xyxy is not None and idx < len(xyxy) and len(xyxy[idx]) >= 4:
                raw_box = (int(xyxy[idx][0]), int(xyxy[idx][1]), int(xyxy[idx][2]), int(xyxy[idx][3]))
            box = clamp_box(raw_box, width, height)
            if box is None:
                continue

            area = (box[2] - box[0]) * (box[3] - box[1])
            vehicles.append((area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

        vehicles.sort(key=lambda item: item[0], reverse=True)
        return human_detected, [(v_type, box) for _, v_type, box in vehicles]

    def detect(
        self,
        image_input: ImageInput,
    ) -> DetectionResult:
        """Execute Stage 1: Detect 4-wheeler vehicles, verify weighbridge occupancy, and extract vehicle crop."""
        pil_img = load_rgb(image_input)
        human_detected, vehicles = self._run_detection(
            pil_img, settings.HUMAN_CONF_THRESH, settings.VEHICLE_CONF_THRESH
        )
        vehicle_count = len(vehicles)
        primary_vehicle_type = vehicles[0][0] if vehicles else None
        primary_vehicle_box = vehicles[0][1] if vehicles else None
        vehicle_crop = pil_img.crop(primary_vehicle_box) if primary_vehicle_box else None

        # Policy 1: Human presence
        if human_detected and settings.REJECT_ON_HUMAN_DETECTED:
            logger.warning("Rejected frame: Human presence detected.")
            return DetectionResult(
                is_eligible=False,
                status=RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
                status_message="Image rejected: Human presence detected.",
                vehicle_detected=vehicle_count > 0,
                vehicle_type=primary_vehicle_type,
                human_detected=human_detected,
                vehicle_count=vehicle_count,
                vehicle_box=primary_vehicle_box,
                crop=vehicle_crop,
            )

        # Policy 2: Multiple vehicles
        if vehicle_count > 1 and settings.REJECT_ON_MULTIPLE_VEHICLES:
            types_str = ", ".join(v[0] for v in vehicles)
            logger.warning(f"Rejected frame: {vehicle_count} vehicles detected ({types_str}).")
            return DetectionResult(
                is_eligible=False,
                status=RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
                status_message=f"Image rejected: Multiple 4-wheeler vehicles detected ({vehicle_count} vehicles: {types_str}). Weighbridge allows only 1 vehicle.",
                vehicle_detected=True,
                vehicle_type=primary_vehicle_type,
                human_detected=human_detected,
                vehicle_count=vehicle_count,
                vehicle_box=primary_vehicle_box,
                crop=vehicle_crop,
            )

        occupancy_note = "with human presence" if human_detected else "with no human occupancy"

        # Policy 3: No vehicle
        if vehicle_count == 0:
            if settings.REJECT_ON_NO_VEHICLE:
                return DetectionResult(
                    is_eligible=False,
                    status=RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
                    status_message="Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.",
                    vehicle_detected=False,
                    vehicle_type=None,
                    human_detected=human_detected,
                    vehicle_count=0,
                    vehicle_box=None,
                    crop=None,
                )
            return DetectionResult(
                is_eligible=True,
                status=None,
                status_message=f"No vehicle detected ({occupancy_note}). Eligible for direct plate recognition.",
                vehicle_detected=False,
                vehicle_type=None,
                human_detected=human_detected,
                vehicle_count=0,
                vehicle_box=None,
                crop=None,
            )

        multi_note = f" ({vehicle_count} vehicles detected)" if vehicle_count > 1 else ""
        return DetectionResult(
            is_eligible=True,
            status=None,
            status_message=f"4-wheeler ({primary_vehicle_type}){multi_note} detected {occupancy_note}. Eligible for plate recognition.",
            vehicle_detected=True,
            vehicle_type=primary_vehicle_type,
            human_detected=human_detected,
            vehicle_count=vehicle_count,
            vehicle_box=primary_vehicle_box,
            crop=vehicle_crop,
        )


def filter_vehicle_and_occupancy(image_input: ImageInput) -> DetectionResult:
    """Convenience function for Stage 1 vehicle detection and occupancy filtering."""
    return VehicleDetector().detect(image_input=image_input)
