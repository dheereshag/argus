import threading
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from PIL import Image

# Import order matters: app.core.config sets the YOLO_CONFIG_DIR environment
# variable, and ultralytics resolves its user config directory at import time.
# If ultralytics is imported first (e.g. via scripts/check_human_gate.py, which
# imports this module before anything else), it falls back to /tmp/Ultralytics
# and logs a spurious warning. Import config first so the env var is set.
from app.core.config import settings
from ultralytics import YOLO

from app.core.contracts import bounded, ensure, require
from app.core.logging import logger
from app.schemas.plate import RecognitionStatusEnum
from app.services.image_processing import clamp_box, load_rgb

class YoloResult(TypedDict, total=True):
    """Typed return value of filter_vehicle_and_occupancy."""
    is_eligible: bool
    status: Optional[Any]                            # RecognitionStatusEnum | None
    status_message: str
    vehicle_detected: bool
    vehicle_type: Optional[str]
    human_detected: bool
    vehicle_box: Optional[Tuple[int, int, int, int]]
    vehicle_count: int


# Global lazy-loaded YOLO model instance, guarded by a lock.
#
# NASA rule 6 (smallest possible scope). This is module-global mutable state
# initialised lazily, and the check-then-set was unsynchronised. That was
# latent while the request handler was async and single-threaded, but the
# handler is now a sync `def` running in FastAPI's threadpool, so two
# concurrent first-requests could both observe None and both construct a YOLO
# model — doubling peak memory at exactly the worst moment, on a Raspberry Pi,
# during startup.
#
# Double-checked locking: the fast path stays lock-free once loaded, and only
# the first callers contend. `main.py` also warms this during lifespan, before
# any request is served, so in practice the lock is belt and braces.
_YOLO_MODEL: Optional[YOLO] = None
_YOLO_LOCK = threading.Lock()


def get_yolo_model() -> YOLO:
    global _YOLO_MODEL  # noqa: PLW0603
    if _YOLO_MODEL is None:
        with _YOLO_LOCK:
            if _YOLO_MODEL is None:  # re-check: another thread may have won the race
                target_model = (
                    settings.YOLO_MODEL_NAME
                    if settings.YOLO_MODEL_NAME and "11" in settings.YOLO_MODEL_NAME
                    else "yolo11n.pt"
                )
                logger.debug(f"Loading YOLO model weights: {target_model}")
                _YOLO_MODEL = YOLO(target_model)
    ensure(_YOLO_MODEL is not None, "YOLO model failed to initialise")
    return _YOLO_MODEL

# COCO Class Names for 4-wheelers
PERSON_CLASS_ID = 0
FOUR_WHEELER_CLASS_NAMES = {
    2: "car",
    5: "bus",
    7: "truck"
}

# Upper bound on detections examined per frame (rule 2). YOLO's own NMS caps at
# 300 by default; a weighbridge frame with more than 100 relevant detections is
# a frame something is wrong with, not one worth iterating fully.
MAX_DETECTIONS = 100

def _run_detection(
    pil_img: Image.Image,
    human_conf_thresh: float,
    vehicle_conf_thresh: float,
) -> Tuple[bool, List[Tuple[str, Tuple[int, int, int, int]]]]:
    """
    Run YOLO and extract (human_present, area-sorted vehicles).

    Split out of filter_vehicle_and_occupancy per rule 4 — that function was
    ~100 lines mixing inference, coordinate handling and policy. Separating
    them means the policy branches can be tested without running a model.

    Every coordinate is clamped to the real frame before leaving this function
    (rule 7). YOLO emits floats that can sit fractionally outside the image, and
    PIL pads out-of-range crops with black rather than raising, so an unclamped
    box silently produces a crop containing invented pixels.
    """
    require(pil_img is not None, "_run_detection called with no image")

    width, height = pil_img.size
    results = get_yolo_model()(pil_img, verbose=False)[0]

    human_detected = False
    vehicles: List[Tuple[int, str, Tuple[int, int, int, int]]] = []

    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False, []

    cls_ids = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else None

    # Bounded: a frame with hundreds of detections is a frame we decline to
    # fully process rather than one we spend unbounded time on (rule 2).
    detections = bounded(list(zip(cls_ids, confs, strict=False)), MAX_DETECTIONS, "YOLO detections")

    for idx, (raw_cls, conf) in enumerate(detections):
        cls_id = int(raw_cls)

        if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh:
            human_detected = True
            continue

        if cls_id not in FOUR_WHEELER_CLASS_NAMES or conf < vehicle_conf_thresh:
            continue

        raw_box = None
        if xyxy is not None and idx < len(xyxy) and len(xyxy[idx]) >= 4:  
            xb = xyxy[idx]
            raw_box = (int(xb[0]), int(xb[1]), int(xb[2]), int(xb[3]))
        box = clamp_box(raw_box, width, height)
        if box is None:
            logger.warning(
                f"[yolo] discarding unusable {FOUR_WHEELER_CLASS_NAMES[cls_id]} box {raw_box} "
                f"for a {width}x{height} frame."
            )
            continue

        area = (box[2] - box[0]) * (box[3] - box[1])
        vehicles.append((area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

    # Largest first: at a weighbridge the vehicle on the platform dominates the
    # frame, and downstream caps keep only the leading few.
    vehicles.sort(key=lambda item: item[0], reverse=True)
    return human_detected, [(v_type, box) for _, v_type, box in vehicles]


def filter_vehicle_and_occupancy(
    image_input: Union[str, bytes],
    human_conf_thresh: Optional[float] = None,
    vehicle_conf_thresh: Optional[float] = None,
) -> YoloResult:
    """
    Pre-screen a frame: is there exactly one 4-wheeler and no person?

    Returns a dict the endpoint maps onto RecognitionResponse. `vehicle_box` is
    guaranteed to be either None or a box already clamped to the frame, so
    callers do not need to re-validate it.
    """
    if human_conf_thresh is None:
        human_conf_thresh = settings.HUMAN_CONF_THRESH
    if vehicle_conf_thresh is None:
        vehicle_conf_thresh = settings.VEHICLE_CONF_THRESH

    require(
        0.0 <= human_conf_thresh <= 1.0,
        f"human_conf_thresh must be in [0, 1], got {human_conf_thresh}",
    )
    require(
        0.0 <= vehicle_conf_thresh <= 1.0,
        f"vehicle_conf_thresh must be in [0, 1], got {vehicle_conf_thresh}",
    )

    pil_img = load_rgb(image_input)
    human_detected, vehicles = _run_detection(pil_img, human_conf_thresh, vehicle_conf_thresh)

    detected_vehicle_types = [v_type for v_type, _ in vehicles]
    vehicle_boxes = [box for _, box in vehicles]
    vehicle_count = len(vehicles)

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
