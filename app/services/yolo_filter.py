import io

from typing import Union, Dict, Any, Tuple
from PIL import Image
from ultralytics import YOLO

from app.core.config import settings
from app.core.logging import logger
from app.schemas.plate import RecognitionStatusEnum

# Global lazy-loaded YOLO model instance
_YOLO_MODEL = None

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        target_model = settings.YOLO_MODEL_NAME if settings.YOLO_MODEL_NAME and "11" in settings.YOLO_MODEL_NAME else "yolo11n.pt"
        logger.debug(f"Loading YOLO model weights: {target_model}")
        _YOLO_MODEL = YOLO(target_model)
    return _YOLO_MODEL

# COCO Class Names for 4-wheelers
PERSON_CLASS_ID = 0
FOUR_WHEELER_CLASS_NAMES = {
    2: "car",
    5: "bus",
    7: "truck"
}

def filter_vehicle_and_occupancy(
    image_input: Union[str, bytes],
    human_conf_thresh: float = None,
    vehicle_conf_thresh: float = None
) -> Dict[str, Any]:
    if human_conf_thresh is None:
        human_conf_thresh = settings.HUMAN_CONF_THRESH
    if vehicle_conf_thresh is None:
        vehicle_conf_thresh = settings.VEHICLE_CONF_THRESH

    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        pil_img = Image.open(image_input).convert("RGB")

    model = get_yolo_model()
    results = model(pil_img, verbose=False)[0]

    human_detected = False
    vehicle_detected = False
    detected_vehicle_type = None
    vehicle_box = None

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy_coords = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") and boxes.xyxy is not None else None

        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            cls_id = int(cls_id)
            if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh:
                human_detected = True
            elif cls_id in FOUR_WHEELER_CLASS_NAMES and conf >= vehicle_conf_thresh:
                vehicle_detected = True
                if not detected_vehicle_type:
                    detected_vehicle_type = FOUR_WHEELER_CLASS_NAMES[cls_id]
                    if xyxy_coords is not None and idx < len(xyxy_coords):
                        vehicle_box = tuple(map(int, xyxy_coords[idx]))

    if human_detected:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
            "status_message": "Image rejected: Human presence detected.",
            "vehicle_detected": vehicle_detected,
            "vehicle_type": detected_vehicle_type,
            "human_detected": human_detected,
            "vehicle_box": vehicle_box
        }

    if not vehicle_detected:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
            "status_message": "Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.",
            "vehicle_detected": False,
            "vehicle_type": None,
            "human_detected": human_detected,
            "vehicle_box": None
        }

    return {
        "is_eligible": True,
        "status": None,
        "status_message": f"4-wheeler ({detected_vehicle_type}) detected with no human occupancy. Eligible for plate recognition.",
        "vehicle_detected": True,
        "vehicle_type": detected_vehicle_type,
        "human_detected": human_detected,
        "vehicle_box": vehicle_box
    }
