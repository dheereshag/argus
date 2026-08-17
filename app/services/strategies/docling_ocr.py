import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.contracts import bounded, require
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import (
    INDIAN_PLATE_REGEX,
    NON_PLATE_WORDS,
    normalize_candidate_strings,
)
from app.services.image_processing import load_rgb

from rapidocr import RapidOCR

# Direct RapidOCR engine instance
_DOCLING_ENGINE = RapidOCR()

# Suppress RapidOCR internal per-crop empty detection warnings on logger and its handlers
_rapid_logger = logging.getLogger("RapidOCR")
_rapid_logger.setLevel(logging.ERROR)
for _h in _rapid_logger.handlers:
    _h.setLevel(logging.ERROR)


def get_docling_engine() -> RapidOCR:
    """Return the RapidOCR engine instance."""
    return _DOCLING_ENGINE


def check_docling_engine() -> bool:
    """Verify that the RapidOCR engine is operational."""
    return _DOCLING_ENGINE is not None


def _get_box_centroid(box: Any) -> Tuple[Optional[float], Optional[float]]:
    """Calculate (x_center, y_center) centroid of an OCR bounding box."""
    if box is None:
        return None, None
    try:
        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
            if all(isinstance(v, (int, float, np.number)) for v in box[:4]):
                return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)
            if all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2 for pt in box):
                pts_x = [pt[0] for pt in box]
                pts_y = [pt[1] for pt in box]
                return float(np.mean(pts_x)), float(np.mean(pts_y))
    except Exception:
        pass
    return None, None


def _is_decal_word(word: str) -> bool:
    """Check if a candidate string is a common commercial vehicle decal word."""
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance local image contrast using CLAHE for low-contrast license plates."""
    np_img = np.array(img)
    if len(np_img.shape) == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


class DoclingStrategy(BasePlateRecognizer):
    """
    Concrete ANPR Strategy using RapidOCR ONNX Runtime engine
    with 2D spatial layout parsing, sub-token word isolation, and positional Indian plate normalisation.
    """

    def __init__(self, **kwargs):
        pass

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> List[Dict[str, Any]]:  # noqa: C901, PLR0912
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        engine = get_docling_engine()
        np_img = np.array(img_pil)

        raw_items: List[Tuple[str, float, Optional[float], Optional[float]]] = []  # (text, score, cx, cy)

        try:
            res = engine(np_img)
            if res is None:
                return []

            # RapidOCR v3.9+ returns RapidOCROutput with .txts, .scores, .boxes
            if hasattr(res, "txts") and hasattr(res, "scores") and res.txts:
                boxes = getattr(res, "boxes", None)
                for idx, (t, s) in enumerate(zip(res.txts, res.scores)):
                    cx, cy = _get_box_centroid(boxes[idx]) if (boxes is not None and idx < len(boxes)) else (None, None)
                    raw_items.append((str(t), float(s), cx, cy))

            elif isinstance(res, (tuple, list)) and len(res) == 2 and isinstance(res[0], (list, tuple)):
                for item in res[0]:
                    if len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        cx, cy = _get_box_centroid(box)
                        raw_items.append((str(text), float(score), cx, cy))

            elif isinstance(res, (tuple, list)):
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        cx, cy = _get_box_centroid(box)
                        raw_items.append((str(text), float(score), cx, cy))
        except Exception as e:
            logger.error(f"[docling/rapidocr] OCR execution failed: {e}")
            return []

        if not raw_items:
            return []

        # Sort items spatially top-to-bottom if coordinates are available
        if any(item[3] is not None for item in raw_items):
            raw_items.sort(key=lambda x: (x[3] if x[3] is not None else 9999.0))

        # Bounded OCR lines
        lines_data = list(bounded(raw_items, settings.MAX_OCR_LINES, "OCR text lines"))

        clean_tokens: List[Tuple[str, Optional[float], Optional[float]]] = []  # (text, cx, cy)
        raw_text_parts: List[str] = []
        seen_tokens = set()

        for text, score, cx, cy in lines_data:
            if not text or score < 0.20:
                continue
            raw_text_parts.append(text.strip())

            # Full line cleaned
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if cleaned and len(cleaned) >= 2 and not _is_decal_word(cleaned):
                if cleaned not in seen_tokens:
                    seen_tokens.add(cleaned)
                    clean_tokens.append((cleaned, cx, cy))

            # Sub-tokens (when a line contains multiple space-separated words)
            for part in text.split():
                part_cleaned = re.sub(r"[^A-Za-z0-9]", "", part).upper()
                if part_cleaned and len(part_cleaned) >= 2 and not _is_decal_word(part_cleaned):
                    if part_cleaned not in seen_tokens:
                        seen_tokens.add(part_cleaned)
                        clean_tokens.append((part_cleaned, cx, cy))

        raw_text_summary = " ".join(raw_text_parts) if raw_text_parts else "N/A"

        # 1. Check individual recognized text tokens
        for cand_raw, _, _ in clean_tokens:
            for cand_norm in normalize_candidate_strings(cand_raw):
                match = INDIAN_PLATE_REGEX.fullmatch(cand_norm)
                if match:
                    info = self.parse_plate_info(match.group(0))
                    if info:
                        info["raw_text"] = raw_text_summary
                        return [info]

        # 2. Check 2-line combinations sorted by 2D spatial proximity
        candidate_pairs: List[Tuple[float, str]] = []
        n = len(clean_tokens)
        for i in range(n):
            tok_a, cx_a, cy_a = clean_tokens[i]
            for j in range(i + 1, min(i + 6, n)):
                tok_b, cx_b, cy_b = clean_tokens[j]
                if cx_a is not None and cy_a is not None and cx_b is not None and cy_b is not None:
                    dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
                else:
                    dist = float(abs(i - j) * 100.0)

                candidate_pairs.append((dist, tok_a + tok_b))
                candidate_pairs.append((dist + 0.1, tok_b + tok_a))

        # Sort candidate pairs by spatial proximity (closest first)
        candidate_pairs.sort(key=lambda p: p[0])

        for _, pair_raw in candidate_pairs:
            for pair_norm in normalize_candidate_strings(pair_raw):
                match = INDIAN_PLATE_REGEX.fullmatch(pair_norm)
                if match:
                    info = self.parse_plate_info(match.group(0))
                    if info:
                        info["raw_text"] = raw_text_summary
                        return [info]

        # 3. Fallback: If no valid plate matched
        return [{
            "plate": "N/A",
            "state": "N/A",
            "raw_text": raw_text_summary
        }]

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """Process an image input with Docling RapidOCR engine, with CLAHE enhancement fallback."""
        pil_img = load_rgb(image_input)
        res = self._extract_plates_from_image_array(pil_img)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
            return res

        # CLAHE Contrast Enhancement Fallback
        try:
            enhanced_img = _enhance_contrast(pil_img)
            res_enh = self._extract_plates_from_image_array(enhanced_img)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res_enh):
                return res_enh
            if res_enh:
                return res_enh
        except Exception as e:
            logger.debug(f"Contrast enhancement fallback skipped: {e}")

        return res
