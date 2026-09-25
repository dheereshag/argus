"""Stage 1: YOLO26 Vehicle Detection, Occupancy Policy Gatekeeper, and Cropper."""

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core import constants
from app.core.contracts import ensure, require
from app.schemas import DetectedVehicle, DetectionResult
from app.services.detector.geometry import BoundingBox, clamp_box, is_contained, pad_box
from app.services.detector.occupancy import partition_humans
from app.services.detector.parser import parse_detections
from app.services.image_processing import ImageInput, load_rgb


class VehicleDetector:
    """Stage 1: YOLO26 Vehicle Detection and Human Localization."""

    _model: YOLO | None = None
    _clamp_box, _pad_box = staticmethod(clamp_box), staticmethod(pad_box)
    _is_contained, _partition_humans = staticmethod(is_contained), staticmethod(partition_humans)
    _parse_detections = staticmethod(parse_detections)

    @classmethod
    def get_model(cls) -> YOLO:
        """Return singleton YOLO26 model instance, loading weights on first access."""
        if cls._model is None:
            import app.services.detector as yf

            cls._model = yf.YOLO(constants.YOLO_MODEL_NAME or "yolo26n.pt")
        ensure(cls._model is not None, "YOLO model failed to initialise")
        return cls._model

    def _run_detection(self, img: Image.Image, h_conf: float, v_conf: float) -> tuple[list[BoundingBox], list[tuple[str, BoundingBox]]]:
        require(img is not None, "_run_detection called with no image")
        res = next(iter(self.get_model()(img, imgsz=constants.DEFAULT_YOLO_IMGSZ, agnostic_nms=constants.YOLO_AGNOSTIC_NMS, verbose=False)))
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cls"):
            return [], []
        c_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        xyxy = (boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)) if getattr(boxes, "xyxy", None) is not None else None
        return parse_detections(c_ids, confs, xyxy, img.width, img.height, h_conf, v_conf)

    @classmethod
    def _build_detected_vehicles(cls, vehicles: list[tuple[str, BoundingBox]], img: Image.Image) -> list[DetectedVehicle]:
        w, h = img.width, img.height
        return [DetectedVehicle(vt, b, img.crop(pad_box(b, w, h)), pad_box(b, w, h)) for vt, b in vehicles]

    def detect(self, image_input: ImageInput) -> DetectionResult:
        """Detect vehicles (car, bus, truck, motorcycle, bicycle), partition humans, and extract crops."""
        pil_img = load_rgb(image_input)
        humans, vehicles = self._run_detection(pil_img, constants.HUMAN_CONF_THRESH, constants.VEHICLE_CONF_THRESH)
        h_out, h_in = partition_humans(humans, vehicles)
        return DetectionResult(self._build_detected_vehicles(vehicles, pil_img), h_out, h_in)
