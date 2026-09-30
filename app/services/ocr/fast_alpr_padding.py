"""Margin expansion utility to prevent edge character truncation."""

from typing import Any

import numpy as np


def compute_padded_crop(
    img: np.ndarray,
    bbox: Any,
    pad_x_pct: float = 0.15,
    pad_y_pct: float = 0.10,
) -> np.ndarray:
    """Expand plate bounding box with margin padding to capture edge characters."""
    h, w = img.shape[:2]
    box_w = bbox.x2 - bbox.x1
    box_h = bbox.y2 - bbox.y1

    px = max(10, int(box_w * pad_x_pct))
    py = max(5, int(box_h * pad_y_pct))

    x1 = max(0, bbox.x1 - px)
    y1 = max(0, bbox.y1 - py)
    x2 = min(w, bbox.x2 + px)
    y2 = min(h, bbox.y2 + py)

    return img[y1:y2, x1:x2]
