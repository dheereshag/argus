"""
Stage 1: YOLO v11 Vehicle Detection, Occupancy Policy Gatekeeper, and Cropper.

This module is responsible for:
  - Loading and caching the YOLO v11 object detection model.
  - Identifying 4-wheeler commercial and passenger vehicles (car, bus, truck) and persons.
  - Enforcing industrial weighbridge occupancy policies (e.g. single-vehicle constraint, no pedestrian presence).
  - Isolating and cropping the primary vehicle bounding box for Stage 2 OCR processing.
"""

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

# Bounding box coordinates: (x_min, y_min, x_max, y_max)
type BoundingBox = tuple[int, int, int, int]


class VehicleDetector:
    """
    Stage 1: YOLO v11 Vehicle Detection, Occupancy Policy Gatekeeper, and Cropper.

    Manages singleton YOLO weights, runs object detection inference on input images,
    filters detections according to weighbridge operational guidelines, and extracts
    the primary vehicle region for downstream license plate recognition.
    """

    _model: YOLO | None = None

    @classmethod
    def get_model(cls) -> YOLO:
        """
        Return the singleton YOLO v11 model instance, loading weights on first call.

        Returns:
            YOLO: Initialized Ultralytics YOLO model.

        Raises:
            ContractViolation: If the model fails to initialize.
        """
        if cls._model is None:
            target_model = settings.YOLO_MODEL_NAME if "11" in (settings.YOLO_MODEL_NAME or "") else "yolo11n.pt"
            logger.debug(f"Loading YOLO model weights: {target_model}")
            cls._model = YOLO(target_model)
        ensure(cls._model is not None, "YOLO model failed to initialise")
        return cls._model

    @staticmethod
    def _clamp_box(box: Any, width: int, height: int) -> BoundingBox | None:
        """
        Clamp an (x1, y1, x2, y2) bounding box to valid image pixel coordinates.

        Performs coordinate validation:
          - Ensures coordinates are ordered such that x1 <= x2 and y1 <= y2.
          - Clamps bounds to [0, width] and [0, height].
          - Rejects boxes where width or height is smaller than MIN_CROP_EDGE_PX.

        Args:
            box: 4-element sequence of coordinate numbers (x1, y1, x2, y2).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            BoundingBox | None: Clamped integer coordinates or None if degenerate/invalid.
        """
        require(width > 0 and height > 0, f"image dimensions must be positive, got {width}x{height}")
        if not box or not isinstance(box, (tuple, list)) or len(box) != 4:
            return None

        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            return None

        # Fix inverted coordinates if model returns swapped corners
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Clamp to real image dimensions
        x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
        x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))

        # Reject degenerate boxes that are too small to contain a readable plate
        if x2 - x1 < MIN_CROP_EDGE_PX or y2 - y1 < MIN_CROP_EDGE_PX:
            return None
        return (x1, y1, x2, y2)

    def _parse_detections(
        self,
        cls_ids: np.ndarray,
        confs: np.ndarray,
        xyxy: np.ndarray | None,
        width: int,
        height: int,
        human_conf_thresh: float,
        vehicle_conf_thresh: float,
    ) -> tuple[bool, list[tuple[str, BoundingBox]]]:
        """Parse raw YOLO output arrays, clamp boxes, and sort 4-wheeler vehicles by area."""
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
            box = self._clamp_box(raw_box, width, height)
            if box is None:
                continue

            area = (box[2] - box[0]) * (box[3] - box[1])
            vehicles.append((area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

        vehicles.sort(key=lambda item: item[0], reverse=True)
        return human_detected, [(v_type, box) for _, v_type, box in vehicles]

    def _run_detection(
        self,
        pil_img: Image.Image,
        human_conf_thresh: float,
        vehicle_conf_thresh: float,
    ) -> tuple[bool, list[tuple[str, BoundingBox]]]:
        """
        Execute YOLO inference on an image and extract filtered detections.

        Parses PyTorch or NumPy output tensors, checks confidence scores against
        configurable thresholds, and sorts detected vehicles by bounding box area
        (descending order) so the dominant vehicle is placed first.
        """
        require(pil_img is not None, "_run_detection called with no image")
        width, height = pil_img.size

        results = next(iter(self.get_model()(pil_img, verbose=False)))
        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cls"):
            return False, []

        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        xyxy = (
            (boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy))
            if getattr(boxes, "xyxy", None) is not None
            else None
        )
        return self._parse_detections(
            cls_ids, confs, xyxy, width, height, human_conf_thresh, vehicle_conf_thresh
        )

    @staticmethod
    def _evaluate_occupancy(
        human_detected: bool,
        vehicles: list[tuple[str, BoundingBox]],
    ) -> tuple[bool, RecognitionStatusEnum | None, str, bool]:
        """
        Evaluate weighbridge occupancy policies based on detection counts and human presence.

        Returns:
            tuple[bool, RecognitionStatusEnum | None, str, bool]:
                (is_eligible, status, status_message, vehicle_detected)
        """
        vehicle_count = len(vehicles)
        primary_vehicle_type = vehicles[0][0] if vehicles else None

        if human_detected and settings.REJECT_ON_HUMAN_DETECTED:
            logger.warning("Rejected frame: Human presence detected.")
            return False, RecognitionStatusEnum.REJECTED_HUMAN_DETECTED, "Image rejected: Human presence detected.", vehicle_count > 0

        if vehicle_count > 1 and settings.REJECT_ON_MULTIPLE_VEHICLES:
            types_str = ", ".join(v[0] for v in vehicles)
            logger.warning(f"Rejected frame: {vehicle_count} vehicles detected ({types_str}).")
            return (
                False,
                RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
                f"Image rejected: Multiple 4-wheeler vehicles detected ({vehicle_count} vehicles: {types_str}). Weighbridge allows only 1 vehicle.",
                True,
            )

        occupancy_note = "with human presence" if human_detected else "with no human occupancy"

        if vehicle_count == 0:
            if settings.REJECT_ON_NO_VEHICLE:
                return False, RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER, "Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.", False
            return True, None, f"No vehicle detected ({occupancy_note}). Eligible for direct plate recognition.", False

        multi_note = f" ({vehicle_count} vehicles detected)" if vehicle_count > 1 else ""
        return True, None, f"4-wheeler ({primary_vehicle_type}){multi_note} detected {occupancy_note}. Eligible for plate recognition.", True

    def detect(self, image_input: ImageInput) -> DetectionResult:
        """
        Execute Stage 1: Detect 4-wheeler vehicles, verify weighbridge occupancy, and extract vehicle crop.

        Evaluates weighbridge business rules in strict priority:
          1. Human Presence: Reject if any person is in the frame (safety & fraud prevention).
          2. Multiple Vehicles: Reject if more than 1 vehicle is present on the scale platform.
          3. Vehicle Presence: Verify at least 1 four-wheeler (car, bus, truck) is localized.

        Args:
            image_input: Input image as file path, raw bytes, PIL Image, or NumPy array.

        Returns:
            DetectionResult: Comprehensive stage 1 outcome with eligibility flag and primary crop.
        """
        pil_img = load_rgb(image_input)
        human_detected, vehicles = self._run_detection(
            pil_img, settings.HUMAN_CONF_THRESH, settings.VEHICLE_CONF_THRESH
        )
        is_eligible, status, status_message, vehicle_detected = self._evaluate_occupancy(
            human_detected, vehicles
        )
        primary_box = vehicles[0][1] if vehicles else None

        return DetectionResult(
            is_eligible=is_eligible,
            status=status,
            status_message=status_message,
            vehicle_detected=vehicle_detected,
            vehicle_type=vehicles[0][0] if vehicles else None,
            human_detected=human_detected,
            vehicle_count=len(vehicles),
            vehicle_box=primary_box,
            crop=pil_img.crop(primary_box) if primary_box else None,
        )
