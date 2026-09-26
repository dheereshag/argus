"""CLAHE contrast enhancement and resolution upscaling for low-contrast images."""

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.logging import logger
from app.services.debug import save_debug_crop


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance image contrast and resolution using CLAHE and bounded linear upscaling."""
    np_img = np.array(img)
    h, w = np_img.shape[:2]

    if min(w, h) < 300:
        scale = min(2.0, 640.0 / max(w, h, 1))
        if scale > 1.05:
            np_img = cv2.resize(np_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    enhanced_rgb = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def retry_contrast(
    recognizer: Any,
    img: Image.Image,
    filename: str = "image.jpg",
    vehicle_idx: int = 0,
) -> list[dict[str, Any]] | None:
    """Attempt contrast-enhanced OCR pass when initial extraction yields no plate."""
    try:
        enhanced = recognizer._enhance_contrast(img)
        save_debug_crop(enhanced, "enhanced", filename, vehicle_idx)
        enh = recognizer._extract_plates_from_image_array(enhanced)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in enh) or enh:
            return enh
    except (cv2.error, ValueError, RuntimeError, OSError, TypeError) as e:
        logger.debug(f"Contrast fallback skipped: {e}")
    return None

