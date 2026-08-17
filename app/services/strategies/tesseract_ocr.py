import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.contracts import bounded, require
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import load_rgb

# Maximum candidate sub-crops examined per ROI crop (NASA Rule 2: fixed loop bounds)
MAX_CANDIDATE_CROPS = 12

# Common OCR letter/digit confusions for Indian license plate positions
_CHAR_TO_DIGIT = {
    "O": "0", "I": "1", "L": "1", "Z": "2", "B": "8",
    "S": "5", "G": "6", "Q": "0", "D": "0"
}
_SERIES_CORRECTIONS = {
    "G3": "GJ", "GT": "GJ", "GI": "GJ", "GB": "GB",
    "D3": "DJ", "DT": "DJ", "DI": "DJ"
}
_STATE_PREFIX_CORRECTIONS = {
    "W8": "WB", "RT": "RJ", "R3": "RJ", "D1": "DL", "D7": "DL", "H8": "HR"
}


def check_tesseract_engine() -> bool:
    """
    Check if pytesseract and system tesseract binary are operational.
    """
    try:
        import pytesseract  # noqa: PLC0415
        _version = pytesseract.get_tesseract_version()
        logger.debug(f"Tesseract OCR engine verified (version {_version}).")
        return True
    except Exception as e:
        logger.warning(f"Tesseract OCR check failed or binary missing: {e}")
        return False


def normalize_candidate_strings(raw_str: str) -> List[str]:
    """
    Generate normalized plate candidate variants using positional character rules for Indian plates.

    Handles common Tesseract OCR character confusions:
      - State code confusions: 'RT' -> 'RJ', 'W8' -> 'WB', 'R3' -> 'RJ'
      - Series code confusions: 'G3' -> 'GJ', 'GT' -> 'GJ', 'GI' -> 'GJ'
      - Positional digit substitutions in district and serial numbers
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned:
        return []

    candidates = [cleaned]

    # State prefix corrections
    for prefix, repl in _STATE_PREFIX_CORRECTIONS.items():
        if cleaned.startswith(prefix):
            candidates.append(repl + cleaned[len(prefix):])

    # Positional substitutions for 10-character plates: [State 2L][District 2D][Series 2L][Serial 4D]
    results = list(candidates)
    for cand in candidates:
        if len(cand) == 10:  
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:6]
            serial = cand[6:10]

            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            series_corr = _SERIES_CORRECTIONS.get(series, series)
            dist_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in dist)
            serial_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series_corr + serial_corr
            if corrected not in results:
                results.append(corrected)

        elif len(cand) == 9:  
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:5]
            serial = cand[5:9]

            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in dist)
            serial_corr = "".join(_CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series + serial_corr
            if corrected not in results:
                results.append(corrected)

    return results


def _preprocess_crop_variants(img_pil: Image.Image) -> List[Any]:
    """
    Apply CLAHE contrast enhancement, dynamic upscaling, and binarization variants.

    Tesseract LSTM performs best on high-contrast text with character heights ~30-35px.
    """
    np_crop = np.array(img_pil)
    gray = cv2.cvtColor(np_crop, cv2.COLOR_RGB2GRAY) if len(np_crop.shape) == 3 else np_crop  
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return [img_pil]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Dynamic scaling targeting readable character height
    scale = max(2.0, min(4.0, 160.0 / max(h, 1)))
    upscaled = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    variants = [upscaled]

    # Otsu thresholding
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Discrete threshold variations for painted/grill lighting variations
    for t_val in (75, 90, 105):
        _, t_bin = cv2.threshold(upscaled, t_val, 255, cv2.THRESH_BINARY)
        variants.append(t_bin)

    return variants


def _extract_plate_candidate_crops(img_pil: Image.Image) -> List[Image.Image]:
    """
    Detect candidate plate sub-regions using horizontal gradient (Sobel-X) and sliding windows.

    Separates the license plate from large vehicle lettering (e.g. 'ASHOK LEYLAND', 'GOODS CARRIER').
    """
    np_img = np.array(img_pil)
    h, w = np_img.shape[:2]
    if h < 16 or w < 32:  
        return [img_pil]

    crops = [img_pil]
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img  

    # 1. Morphological horizontal text clustering
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    blurred = cv2.GaussianBlur(abs_grad_x, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(max(ch, 1))
        area = cw * ch
        if 1.1 <= aspect <= 6.5 and (0.005 * w * h) <= area <= (0.40 * w * h) and ch >= 12 and cw >= 25:  
            pad_x = int(cw * 0.20)
            pad_y = int(ch * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            crops.append(img_pil.crop((x1, y1, x2, y2)))

    # 2. Sliding window crops across bumper area (lower 70%)
    bumper_h = int(h * 0.70)
    win_w = int(w * 0.45)
    step_x = max(1, int(w * 0.15))
    for bx in range(0, max(1, w - int(w * 0.35)), step_x):
        crops.append(img_pil.crop((bx, h - bumper_h, min(w, bx + win_w), h)))

    return list(bounded(crops, MAX_CANDIDATE_CROPS, "candidate plate crops"))


class TesseractStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using Tesseract OCR (via pytesseract) for local license plate text extraction.
    Features adaptive CLAHE preprocessing, dynamic resolution scaling, and candidate plate extraction.
    """

    def __init__(self, bottom_crop_ratio: float = 0.50):
        self.bottom_crop_ratio = bottom_crop_ratio

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> List[Dict[str, Any]]:  # noqa: C901, PLR0912
        require(img_pil is not None, "_extract_plates_from_image_array received None")

        import pytesseract  # noqa: PLC0415

        candidate_crops = _extract_plate_candidate_crops(img_pil)
        ocr_configs = (
            f"--psm {settings.TESSERACT_PSM} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            f"--psm {settings.TESSERACT_PSM}",
        )

        all_clean_lines: List[str] = []

        for crop in candidate_crops:
            variants = _preprocess_crop_variants(crop)
            for var in variants:
                for cfg in ocr_configs:
                    try:
                        raw_text = pytesseract.image_to_string(var, config=cfg, lang=settings.TESSERACT_LANG)
                    except Exception as e:
                        logger.error(f"[tesseract] Failed to perform OCR: {e}")
                        return []

                    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                    if not raw_lines:
                        continue

                    lines = list(bounded(raw_lines, settings.MAX_OCR_LINES, "OCR text lines"))
                    clean_lines: List[str] = []
                    for line in lines:
                        cleaned = re.sub(r"[^A-Za-z0-9]", "", line).upper()
                        if cleaned:
                            clean_lines.append(cleaned)

                    if clean_lines:
                        all_clean_lines.extend(clean_lines)

                    raw_text_summary = " ".join(clean_lines) if clean_lines else "N/A"

                    # 1. Inspect individual text lines
                    for cand_raw in clean_lines:
                        for cand_norm in normalize_candidate_strings(cand_raw):
                            match = INDIAN_PLATE_REGEX.fullmatch(cand_norm)
                            if match:
                                info = self.parse_plate_info(match.group(0))
                                if info:
                                    info["raw_text"] = raw_text_summary
                                    return [info]

                    # 2. Inspect adjacent and near-adjacent text line pairs (for 2-line Indian plates)
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

        # 3. Fallback: If no valid Indian plate matched, return best detected text
        fallback_summary = " ".join(dict.fromkeys(all_clean_lines[:10])) if all_clean_lines else "N/A"
        return [{
            "plate": "N/A",
            "state": "N/A",
            "raw_text": fallback_summary
        }]

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """Process a single image crop or full image with Tesseract OCR."""
        pil_img = load_rgb(image_input)
        return self._extract_plates_from_image_array(pil_img)
