"""Stage 1: YOLO26 Vehicle Detection and Crop Extraction."""

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core import constants
from app.core.contracts import ensure, require
from app.schemas import DetectedVehicle, DetectionResult
from app.services.detector.geometry import BoundingBox, clamp_box, pad_box
from app.services.detector.parser import parse_detections
from app.services.image_processing import ImageInput, load_rgb


class VehicleDetector:
    """Stage 1: YOLO26 Vehicle Detection."""

    _model: YOLO | None = None
    _clamp_box, _pad_box = staticmethod(clamp_box), staticmethod(pad_box)
    _parse_detections = staticmethod(parse_detections)

    @classmethod
    def get_model(cls) -> YOLO:
        if cls._model is None:
            import app.services.detector as yf

            cls._model = yf.YOLO(constants.YOLO_MODEL_NAME or "yolo26n.pt")
        ensure(cls._model is not None, "YOLO model failed to initialise")
        return cls._model

    def _run_detection(self, img: Image.Image, v_conf: float) -> list[tuple[str, BoundingBox]]:
        require(img is not None, "_run_detection called with no image")
        res = next(iter(self.get_model()(img, imgsz=constants.DEFAULT_YOLO_IMGSZ, agnostic_nms=constants.YOLO_AGNOSTIC_NMS, verbose=False)))
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cls"):
            return []
        c_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        xyxy = (boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)) if getattr(boxes, "xyxy", None) is not None else None
        _, vehicles = parse_detections(c_ids, confs, xyxy, img.width, img.height, 1.0, v_conf)
        return vehicles

    def detect(self, image_input: ImageInput) -> DetectionResult:
        pil_img = load_rgb(image_input)
        vehicles = self._run_detection(pil_img, constants.VEHICLE_CONF_THRESH)
        w, h = pil_img.width, pil_img.height
        detected = [DetectedVehicle(vt, b, pil_img.crop(pad_box(b, w, h)), pad_box(b, w, h)) for vt, b in vehicles]
        return DetectionResult(vehicles=detected)
