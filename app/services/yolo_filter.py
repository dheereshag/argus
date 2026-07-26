import io
from typing import Union, Dict, Any, Tuple
from PIL import Image
from ultralytics import YOLO

from app.schemas.plate import RecognitionStatusEnum

# Global lazy-loaded YOLO model instance
_YOLO_MODEL = None

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        # Load lightweight YOLO model
        try:
            _YOLO_MODEL = YOLO("yolo11n.pt")
        except Exception:
            _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL

# COCO Class IDs
PERSON_CLASS_ID = 0
FOUR_WHEELER_CLASS_IDS = {2, 5, 7}  # 2: car, 5: bus, 7: truck

def filter_vehicle_and_occupancy(
    image_input: Union[str, bytes],
    human_conf_thresh: float = 0.30,
    vehicle_conf_thresh: float = 0.35
) -> Dict[str, Any]:
    """
    YOLO pre-screening function to verify:
    1. A 4-wheeler (car, bus, truck) is present.
    2. No human (person) is present in the image.

    Returns dict with keys:
    - is_eligible: bool
    - status: RecognitionStatusEnum (if ineligible)
    - status_message: str
    - vehicle_detected: bool
    - human_detected: bool
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        pil_img = Image.open(image_input).convert("RGB")

    model = get_yolo_model()
    results = model(pil_img, verbose=False)[0]

    human_detected = False
    vehicle_detected = False

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for cls_id, conf in zip(cls_ids, confs):
            cls_id = int(cls_id)
            if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh:
                human_detected = True
            elif cls_id in FOUR_WHEELER_CLASS_IDS and conf >= vehicle_conf_thresh:
                vehicle_detected = True

    if human_detected:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
            "status_message": "Image rejected: Human presence detected.",
            "vehicle_detected": vehicle_detected,
            "human_detected": human_detected
        }

    if not vehicle_detected:
        return {
            "is_eligible": False,
            "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
            "status_message": "Image rejected: No 4-wheeler vehicle (car, bus, truck) detected.",
            "vehicle_detected": vehicle_detected,
            "human_detected": human_detected
        }

    return {
        "is_eligible": True,
        "status": None,
        "status_message": "4-wheeler detected with no human occupancy. Eligible for plate recognition.",
        "vehicle_detected": vehicle_detected,
        "human_detected": human_detected
    }
