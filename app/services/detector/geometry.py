"""Bounding box coordinate validation, clamping, padding, and containment math."""

from typing import Any

from app.constants import MIN_CROP_EDGE_PX
from app.core.contracts import require

type BoundingBox = tuple[int, int, int, int]


def clamp_box(box: Any, width: int, height: int) -> BoundingBox | None:
    """Clamp an (x1, y1, x2, y2) bounding box to valid image pixel coordinates."""
    require(width > 0 and height > 0, f"dimensions must be positive, got {width}x{height}")
    if not box or not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1, y1, x2, y2 = max(0, min(x1, width)), max(0, min(y1, height)), max(0, min(x2, width)), max(0, min(y2, height))

    if x2 - x1 < MIN_CROP_EDGE_PX or y2 - y1 < MIN_CROP_EDGE_PX:
        return None
    return (x1, y1, x2, y2)


def pad_box(box: BoundingBox, width: int, height: int, padding_ratio: float = 0.03) -> BoundingBox:
    """Add a safety padding margin around a bounding box, clamped to image boundaries."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    return (max(0, box[0] - pad_x), max(0, box[1] - pad_y), min(width, box[2] + pad_x), min(height, box[3] + pad_y))


def is_contained(inner: BoundingBox, outer: BoundingBox, threshold: float = 0.80) -> bool:
    """Check if inner box is predominantly contained within outer box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (inter_area / inner_area) >= threshold


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1, union)

