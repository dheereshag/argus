"""Execute Fast-ALPR on full frames and vehicle crops with Indian plate validation."""

import statistics
from typing import Any

import cv2
import numpy as np

from app.services.image_processing import ImageInput, load_rgb
from app.services.ocr.fast_alpr_engine import get_fast_alpr_engine
from app.services.ocr.fast_alpr_padding import compute_padded_crop
from app.services.plate_rules import resolve_plate_from_text


def _calc_conf(ocr_res: Any) -> float:
    if not ocr_res or not ocr_res.confidence:
        return 0.0
    if isinstance(ocr_res.confidence, list):
        return statistics.mean(ocr_res.confidence) if ocr_res.confidence else 0.0
    return float(ocr_res.confidence)


def run_fast_alpr(img_bgr: np.ndarray) -> list[str]:
    """Run Fast-ALPR on a BGR image/crop, sort by confidence, validate Indian plates."""
    alpr = get_fast_alpr_engine()
    detections = alpr.detector.predict(img_bgr)
    if not detections:
        return []
    candidates: list[tuple[float, str]] = []
    for det in detections:
        crop = compute_padded_crop(img_bgr, det.bounding_box)
        ocr_res = alpr.ocr.predict(crop)
        if ocr_res and ocr_res.text:
            candidates.append((_calc_conf(ocr_res), ocr_res.text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    valid_plates: list[str] = []
    for _, raw_text in candidates:
        if (p := resolve_plate_from_text(raw_text)) and p not in valid_plates:
            valid_plates.append(p)
    return valid_plates


def run_fast_alpr_pipeline(image_input: ImageInput) -> list[str]:
    """Tier 1: Run Fast-ALPR directly on full input image."""
    pil_img = load_rgb(image_input)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return run_fast_alpr(img_bgr)


def run_fast_alpr_on_crops(vehicles: list[Any]) -> list[str]:
    """Tier 2: Run Fast-ALPR on cropped vehicle boxes."""
    for v in vehicles:
        if v.crop is None:
            continue
        crop_bgr = cv2.cvtColor(np.array(v.crop), cv2.COLOR_RGB2BGR)
        if plates := run_fast_alpr(crop_bgr):
            return plates
    return []
