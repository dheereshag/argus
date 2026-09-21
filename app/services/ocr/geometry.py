"""Geometry parsing and centroid calculations for OCR bounding boxes."""

from typing import Any

import numpy as np


def get_box_geometry(box: Any) -> tuple[float | None, float | None, tuple[int, int, int, int] | None]:
    """Extract (cx, cy, (x1, y1, x2, y2)) from RapidOCR bounding box or polygon."""
    if box is None:
        return None, None, None
    try:
        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
            if all(isinstance(v, (int, float, np.number)) for v in box[:4]):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), (x1, y1, x2, y2)
            if all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2 for pt in box):
                pts_x = [float(pt[0]) for pt in box]
                pts_y = [float(pt[1]) for pt in box]
                return float(np.mean(pts_x)), float(np.mean(pts_y)), (int(min(pts_x)), int(min(pts_y)), int(max(pts_x)), int(max(pts_y)))
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    return None, None, None


def get_box_centroid(box: Any) -> tuple[float | None, float | None]:
    """Calculate the (x_center, y_center) centroid of an OCR bounding box."""
    cx, cy, _ = get_box_geometry(box)
    return cx, cy
