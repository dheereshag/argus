import io

from typing import Union, Dict, Any, Tuple
from PIL import Image, ImageOps
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
        pil_img = Image.open(io.BytesIO(image_input))
    else:
        pil_img = Image.open(image_input)

    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

    model = get_yolo_model()
    results = model(pil_img, verbose=False)[0]

    human_detected = False
    vehicle_count = 0
    detected_vehicle_types = []
    vehicle_boxes = []

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
                vehicle_count += 1
                v_type = FOUR_WHEELER_CLASS_NAMES[cls_id]
                detected_vehicle_types.append(v_type)
                if xyxy_coords is not None and idx < len(xyxy_coords):
                    vehicle_boxes.append(tuple(map(int, xyxy_coords[idx])))

    primary_vehicle_type = detected_vehicle_types[0] if detected_vehicle_types else None
    primary_vehicle_box = vehicle_boxes[0] if vehicle_boxes else None

    if human_detected:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
            "status_message": "Image rejected: Human presence detected.",
            "vehicle_detected": vehicle_count > 0,
            "vehicle_type": primary_vehicle_type,
            "human_detected": human_detected,
            "vehicle_box": primary_vehicle_box,
            "vehicle_count": vehicle_count
        }

    if vehicle_count > 1:
        types_str = ", ".join(detected_vehicle_types)
        logger.warning(f"Rejected frame: {vehicle_count} vehicles detected ({types_str}).")
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
            "status_message": f"Image rejected: Multiple 4-wheeler vehicles detected ({vehicle_count} vehicles: {types_str}). Weighbridge allows only 1 vehicle.",
            "vehicle_detected": True,
            "vehicle_type": primary_vehicle_type,
            "human_detected": human_detected,
            "vehicle_box": primary_vehicle_box,
            "vehicle_count": vehicle_count
        }

    if vehicle_count == 0:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
            "status_message": "Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.",
            "vehicle_detected": False,
            "vehicle_type": None,
            "human_detected": human_detected,
            "vehicle_box": None,
            "vehicle_count": 0
        }

    return {
        "is_eligible": True,
        "status": None,
        "status_message": f"4-wheeler ({primary_vehicle_type}) detected with no human occupancy. Eligible for plate recognition.",
        "vehicle_detected": True,
        "vehicle_type": primary_vehicle_type,
        "human_detected": human_detected,
        "vehicle_box": primary_vehicle_box,
        "vehicle_count": 1
    }
