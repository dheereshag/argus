"""Parse raw YOLO output arrays, clamp boxes, count humans, and sort vehicles."""

import numpy as np

from app.constants import FOUR_WHEELER_CLASS_NAMES, MAX_DETECTIONS, PERSON_CLASS_ID
from app.core.config import settings
from app.core.contracts import bounded
from app.services.detector.geometry import BoundingBox, clamp_box, is_contained


def parse_detections(
    cls_ids: np.ndarray,
    confs: np.ndarray,
    xyxy: np.ndarray | None,
    width: int,
    height: int,
    human_conf_thresh: float,
    vehicle_conf_thresh: float,
) -> tuple[int, list[tuple[str, BoundingBox]]]:
    """Parse raw YOLO output arrays, clamp boxes, count humans, and sort vehicles."""
    human_candidates: list[BoundingBox] = []
    vehicles: list[tuple[int, str, BoundingBox]] = []
    total_area = width * height
    min_h_area = settings.MIN_HUMAN_BOX_AREA_RATIO * total_area
    min_v_area = settings.MIN_VEHICLE_BOX_AREA_RATIO * total_area

    for idx, (raw_cls, conf) in enumerate(bounded(list(zip(cls_ids, confs, strict=False)), MAX_DETECTIONS, "detections")):
        cls_id = int(raw_cls)
        raw_b = (int(xyxy[idx][0]), int(xyxy[idx][1]), int(xyxy[idx][2]), int(xyxy[idx][3])) if xyxy is not None and idx < len(xyxy) and len(xyxy[idx]) >= 4 else None
        box = clamp_box(raw_b, width, height)
        if box is None:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])

        if cls_id == PERSON_CLASS_ID and conf >= human_conf_thresh and area >= min_h_area:
            human_candidates.append(box)
        elif cls_id in FOUR_WHEELER_CLASS_NAMES and conf >= vehicle_conf_thresh and area >= min_v_area:
            vehicles.append((area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

    vehicles.sort(key=lambda item: item[0], reverse=True)
    sorted_v = [(v_type, box) for _, v_type, box in vehicles]

    if settings.ALLOW_CAB_OCCUPANTS and sorted_v:
        human_count = sum(1 for hb in human_candidates if not any(is_contained(hb, vb) for _, vb in sorted_v))
    else:
        human_count = len(human_candidates)

    return human_count, sorted_v
