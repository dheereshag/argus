import re
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Singleton RapidOCR engine instance
_DOCLING_ENGINE = None


def get_docling_engine() -> Any:
    """
    Get or initialize the RapidOCR engine singleton.
    """
    global _DOCLING_ENGINE  # noqa: PLW0603
    if _DOCLING_ENGINE is None:
        try:
            from rapidocr import RapidOCR  # noqa: PLC0415
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        _DOCLING_ENGINE = RapidOCR()
        logger.info("Docling RapidOCR engine initialized successfully.")
    return _DOCLING_ENGINE


def check_docling_engine() -> bool:
    """
    Verify that the Docling / RapidOCR engine is operational.
    """
    try:
        engine = get_docling_engine()
        return engine is not None
    except Exception as e:
        logger.warning(f"Docling OCR check failed: {e}")
        return False


def _get_box_y_center(box: Any) -> Optional[float]:
    """Calculate vertical center coordinate of an OCR bounding box for spatial layout sorting."""
    if box is None:
        return None
    try:
        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
            if all(isinstance(v, (int, float, np.number)) for v in box[:4]):
                return float((box[1] + box[3]) / 2.0)
            if all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2 for pt in box):
                return float(np.mean([pt[1] for pt in box]))
    except Exception:
        pass
    return None


def _is_decal_word(word: str) -> bool:
    """Check if a candidate string is a common commercial vehicle decal word."""
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


class DoclingStrategy(BasePlateRecognizer):
    """
    Concrete ANPR Strategy using RapidOCR ONNX Runtime engine
    with spatial layout parsing and positional Indian plate normalisation.
    """

    def __init__(self, **kwargs):
        pass

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> List[Dict[str, Any]]:  # noqa: C901, PLR0912
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        engine = get_docling_engine()
        np_img = np.array(img_pil)

        raw_items: List[Tuple[str, float, Optional[float]]] = []  # (text, score, y_center)

        try:
            res = engine(np_img)
            if res is None:
                return []

            # RapidOCR v3.9+ returns RapidOCROutput with .txts, .scores, .boxes
            if hasattr(res, "txts") and hasattr(res, "scores") and res.txts:
                boxes = getattr(res, "boxes", None)
                for idx, (t, s) in enumerate(zip(res.txts, res.scores)):
                    y_center = _get_box_y_center(boxes[idx]) if (boxes is not None and idx < len(boxes)) else None
                    raw_items.append((str(t), float(s), y_center))

            elif isinstance(res, (tuple, list)) and len(res) == 2 and isinstance(res[0], (list, tuple)):
                for item in res[0]:
                    if len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        y_center = _get_box_y_center(box)
                        raw_items.append((str(text), float(score), y_center))

            elif isinstance(res, (tuple, list)):
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        box, text, score = item[0], item[1], item[2]
                        y_center = _get_box_y_center(box)
                        raw_items.append((str(text), float(score), y_center))
        except Exception as e:
            logger.error(f"[docling/rapidocr] OCR execution failed: {e}")
            return []

        if not raw_items:
            return []

        # Sort items spatially top-to-bottom if coordinates are available
        if any(item[2] is not None for item in raw_items):
            raw_items.sort(key=lambda x: (x[2] if x[2] is not None else 9999))

        # Bounded OCR lines
        lines_data = list(bounded(raw_items, settings.MAX_OCR_LINES, "OCR text lines"))

        clean_lines: List[str] = []
        raw_text_parts: List[str] = []

        for text, score, _ in lines_data:
            if not text or score < 0.20:
                continue
            raw_text_parts.append(text.strip())
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if cleaned and len(cleaned) >= 2 and not _is_decal_word(cleaned):
                clean_lines.append(cleaned)

        raw_text_summary = " ".join(raw_text_parts) if raw_text_parts else "N/A"

        # 1. Check individual recognized text lines
        for cand_raw in clean_lines:
            for cand_norm in normalize_candidate_strings(cand_raw):
                match = INDIAN_PLATE_REGEX.fullmatch(cand_norm)
                if match:
                    info = self.parse_plate_info(match.group(0))
                    if info:
                        info["raw_text"] = raw_text_summary
                        return [info]

        # 2. Check 2-line adjacent and near-adjacent combinations (stacked plates)
        n = len(clean_lines)
        for i in range(n):
            for j in (i + 1, i + 2):
                if j < n:
                    pair_raw = clean_lines[i] + clean_lines[j]
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
        """Process an image input with Docling RapidOCR engine."""
        pil_img = load_rgb(image_input)
        return self._extract_plates_from_image_array(pil_img)
