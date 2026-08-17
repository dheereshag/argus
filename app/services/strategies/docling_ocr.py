import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.contracts import bounded, require
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import load_rgb

# Singleton RapidOCR engine instance
_DOCLING_ENGINE = None


def get_docling_engine() -> Any:
    """
    Get or initialize the RapidOCR engine singleton.
    """
    global _DOCLING_ENGINE
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


# Positional character confusions for Indian license plates
_CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

_DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

_SERIES_CORRECTIONS = {
    "G3": "GJ", "GT": "GJ", "GI": "GJ", "GB": "GB",
    "D3": "DJ", "DT": "DJ", "DI": "DJ"
}

_STATE_PREFIX_CORRECTIONS = {
    "W8": "WB", "RT": "RJ", "R3": "RJ", "D1": "DL", "D7": "DL", "H8": "HR", "AS": "AS"
}


def normalize_candidate_strings(raw_str: str) -> List[str]:
    """
    Generate normalized plate candidate variants using positional character rules for Indian plates.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 6:
        return []

    candidates = [cleaned]

    # State prefix corrections
    for prefix, repl in _STATE_PREFIX_CORRECTIONS.items():
        if cleaned.startswith(prefix):
            candidates.append(repl + cleaned[len(prefix):])

    results = list(candidates)
    for cand in candidates:
        # Standard 10-character plate: [State 2L][District 2D][Series 2L][Serial 4D]
        if len(cand) == 10:
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:6]
            serial = cand[6:10]

            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in dist)
            series_corr = _SERIES_CORRECTIONS.get(series, "".join(_DIGIT_TO_CHAR.get(c, c) for c in series))
            serial_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series_corr + serial_corr
            if corrected not in results:
                results.append(corrected)

        # 9-character plate: [State 2L][District 2D][Series 1L][Serial 4D]
        elif len(cand) == 9:
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:5]
            serial = cand[5:9]

            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in dist)
            series_corr = "".join(_DIGIT_TO_CHAR.get(c, c) for c in series)
            serial_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series_corr + serial_corr
            if corrected not in results:
                results.append(corrected)

            # Single-digit district for DL (e.g. DL1CX2744)
            dist_1 = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_2 = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[3:5])
            ser_4 = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_alt = st_corr + dist_1 + ser_2 + ser_4
            if cand_alt not in results:
                results.append(cand_alt)

        # 8-character plate
        elif len(cand) == 8:
            st = cand[:2]
            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_a = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_a = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[3:4])
            num_a = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[4:8])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            dist_b = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_b = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_b = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[5:8])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        # Bharat (BH) series
        if "BH" in cand:
            idx = cand.find("BH")
            if idx >= 2 and len(cand) >= idx + 6:
                yr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[idx-2:idx])
                serial = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[idx+2:idx+6])
                ser = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[idx+6:])
                bh_cand = yr + "BH" + serial + ser
                if bh_cand not in results:
                    results.append(bh_cand)

    return results


class DoclingStrategy(BasePlateRecognizer):
    """
    Concrete ANPR Strategy using Docling RapidOCR (ONNX Runtime engine)
    for fast, high-accuracy license plate character recognition.
    """

    def __init__(self, **kwargs):
        pass

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> List[Dict[str, Any]]:  # noqa: C901, PLR0912
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        engine = get_docling_engine()
        np_img = np.array(img_pil)

        raw_items: List[Tuple[str, float]] = []

        try:
            res = engine(np_img)
            if res is None:
                return []

            # RapidOCR v3.9+ returns RapidOCROutput with .txts, .scores, .boxes
            if hasattr(res, "txts") and hasattr(res, "scores") and res.txts:
                raw_items = [(str(t), float(s)) for t, s in zip(res.txts, res.scores)]
            elif isinstance(res, (tuple, list)) and len(res) == 2 and isinstance(res[0], (list, tuple)):
                raw_items = [(str(item[1]), float(item[2])) for item in res[0] if len(item) >= 3]
            elif isinstance(res, (tuple, list)):
                raw_items = [(str(item[1]), float(item[2])) for item in res if isinstance(item, (list, tuple)) and len(item) >= 3]
        except Exception as e:
            logger.error(f"[docling/rapidocr] OCR execution failed: {e}")
            return []

        if not raw_items:
            return []

        # Bounded OCR lines
        lines_data = list(bounded(raw_items, settings.MAX_OCR_LINES, "OCR text lines"))

        clean_lines: List[str] = []
        raw_text_parts: List[str] = []

        for text, score in lines_data:
            if not text or score < 0.20:
                continue
            raw_text_parts.append(text.strip())
            cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if cleaned and len(cleaned) >= 2:
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
