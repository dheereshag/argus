"""CLAHE contrast enhancement and resolution upscaling for low-contrast images."""

from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.logging import logger


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance image contrast and resolution using CLAHE, Black-Hat, unsharp mask, and bicubic upscaling."""
    np_img = np.array(img)
    h, w = np_img.shape[:2]

    if min(w, h) < 300:
        scale = min(2.0, 640.0 / max(w, h, 1))
        if scale > 1.05:
            np_img = cv2.resize(np_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    k_size = max(5, int(min(w, h) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    blackhat = cv2.morphologyEx(cl, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.subtract(cl, blackhat)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)
    enhanced_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def retry_contrast(recognizer: Any, img: Image.Image) -> list[dict[str, Any]] | None:
    """Attempt contrast-enhanced OCR pass when initial extraction yields no plate."""
    try:
        enh = recognizer._extract_plates_from_image_array(recognizer._enhance_contrast(img))
        if any(r.get("plate") and r.get("plate") != "N/A" for r in enh) or enh:
            return enh
    except (cv2.error, ValueError, RuntimeError, OSError, TypeError) as e:
        logger.debug(f"Contrast fallback skipped: {e}")
    return None
