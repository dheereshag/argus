"""Parse raw YOLO output arrays, clamp boxes, count humans, and sort vehicles."""

import numpy as np

from app.constants import FOUR_WHEELER_CLASS_NAMES, MAX_DETECTIONS, PERSON_CLASS_ID
from app.core.config import settings
from app.core.contracts import bounded
from app.services.detector.geometry import BoundingBox, box_iou, clamp_box, is_contained


def _dedup_vehicles(candidates: list[tuple[float, int, str, BoundingBox]]) -> list[tuple[str, BoundingBox]]:
    """Deduplicate overlapping vehicles by confidence and spatial IoU."""
    candidates.sort(key=lambda item: item[0], reverse=True)
    deduped: list[tuple[str, BoundingBox]] = []
    thresh = settings.VEHICLE_IOU_THRESH
    for _, _, v_type, b in candidates:
        if not any(box_iou(b, eb) > thresh or is_contained(b, eb, 0.70) or is_contained(eb, b, 0.70) for _, eb in deduped):
            deduped.append((v_type, b))
    deduped.sort(key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)
    return deduped


def parse_detections(
    cls_ids: np.ndarray,
    confs: np.ndarray,
    xyxy: np.ndarray | None,
    width: int,
    height: int,
    human_conf_thresh: float,
    vehicle_conf_thresh: float,
) -> tuple[list[BoundingBox], list[tuple[str, BoundingBox]]]:
    """Parse raw YOLO output arrays, clamp boxes, filter candidates, and sort vehicles."""
    human_candidates: list[BoundingBox] = []
    vehicle_candidates: list[tuple[float, int, str, BoundingBox]] = []
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
            vehicle_candidates.append((float(conf), area, FOUR_WHEELER_CLASS_NAMES[cls_id], box))

    vehicles = _dedup_vehicles(vehicle_candidates)
    return human_candidates, vehicles

