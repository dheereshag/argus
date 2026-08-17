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

# Common OCR letter/digit confusions for Indian license plate positions
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
      - State prefix confusions: 'RT' -> 'RJ', 'W8' -> 'WB', 'R3' -> 'RJ', 'D1' -> 'DL', 'H8' -> 'HR'
      - District code digit substitutions (positions 3-4): 'O/D' -> '0', 'I/L' -> '1', 'B' -> '8', 'S' -> '5', 'G' -> '6', 'A' -> '4'
      - Series letter substitutions (positions 5-6): '0' -> 'O', '1' -> 'I', '8' -> 'B', '5' -> 'S', '6' -> 'G', '4' -> 'A'
      - Serial number digit substitutions (positions 7-10): 'O/D' -> '0', 'I/L' -> '1', 'B' -> '8', 'S' -> '5', 'G' -> '6', 'A' -> '4'
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
        # e.g. MH06TJ8664, RJ14GJ4976, DL01AB1234
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

        # 9-character plate with 1-letter series: [State 2L][District 2D][Series 1L][Serial 4D]
        # e.g. KA25B3155, RJ09G4017, MH15C2859
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

            # Also test single-digit district: [State 2L][District 1D][Series 2L][Serial 4D] (e.g. DL1CX2744)
            dist_1 = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_2 = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[3:5])
            ser_4 = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_alt = st_corr + dist_1 + ser_2 + ser_4
            if cand_alt not in results:
                results.append(cand_alt)

        # 8-character plate: [State 2L][District 1D][Series 1L][Serial 4D] or [State 2L][District 2D][Series 1L][Serial 3D]
        elif len(cand) == 8:
            st = cand[:2]
            st_corr = _STATE_PREFIX_CORRECTIONS.get(st, st)
            # Case A: DL 1 C 2744
            dist_a = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_a = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[3:4])
            num_a = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[4:8])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            # Case B: RJ 40 G 315
            dist_b = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_b = "".join(_DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_b = "".join(_CHAR_TO_DIGIT.get(c, c) for c in cand[5:8])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        # Bharat (BH) series: [Year 2D][BH][Serial 4D][Series 1-2L]
        # e.g. 22BH1234AA
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


def _preprocess_crop_variants(img_pil: Image.Image) -> List[Any]:
    """
    Apply CLAHE contrast enhancement, character height scaling, and clean thresholding.

    Tesseract LSTM performs best on high-contrast black-on-white text with
    character heights ~30-35px (plate height ~60-80px) and a white border.
    """
    np_crop = np.array(img_pil)
    gray = cv2.cvtColor(np_crop, cv2.COLOR_RGB2GRAY) if len(np_crop.shape) == 3 else np_crop
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return [img_pil]

    # Target standard plate height ~70px (character height ~35px)
    target_h = 70.0
    scale = max(1.0, min(4.0, target_h / max(h, 1)))
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Bilateral smoothing to remove noise while preserving character edges
    smoothed = cv2.bilateralFilter(resized, 9, 15, 15)

    # 10px white border for Tesseract
    bordered = cv2.copyMakeBorder(smoothed, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bordered)

    variants = [enhanced]

    # Otsu thresholding
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Inverted Otsu (for white/yellow text on dark plates or dark background)
    otsu_inv = cv2.bitwise_not(otsu)
    variants.append(otsu_inv)

    # Adaptive Gaussian threshold
    adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(adapt)

    return variants


def _extract_plate_candidate_crops(img_pil: Image.Image) -> List[Image.Image]:
    """
    Detect candidate plate sub-regions using Top-Hat / Black-Hat morphology and bumper priors.
    """
    np_img = np.array(img_pil)
    h, w = np_img.shape[:2]
    if h < 16 or w < 32:
        return [img_pil]

    crops: List[Image.Image] = []
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img

    # 1. Bumper area priors (lower 40% and lower 60%)
    crops.append(img_pil.crop((0, int(h * 0.60), w, h)))
    crops.append(img_pil.crop((0, int(h * 0.40), w, h)))

    # 2. Morphological Top-Hat and Black-Hat gradient to isolate plate text rectangles
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, rect_kernel)
    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, rect_kernel)
    grad = cv2.add(tophat, blackhat)

    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_rects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(max(ch, 1))
        area = cw * ch
        # Rectangular plates (aspect 2.0-6.5) or 2-line stacked plates (aspect 1.1-2.2)
        if 1.1 <= aspect <= 6.5 and (0.003 * w * h) <= area <= (0.35 * w * h) and ch >= 12 and cw >= 25:
            pad_x = int(cw * 0.15)
            pad_y = int(ch * 0.20)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            candidate_rects.append((x1, y1, x2, y2, area))

    # Sort largest area first
    candidate_rects.sort(key=lambda b: b[4], reverse=True)
    for x1, y1, x2, y2, _ in candidate_rects:
        crops.append(img_pil.crop((x1, y1, x2, y2)))

    # 3. Full crop as fallback
    crops.append(img_pil)

    return list(bounded(crops, settings.MAX_CANDIDATE_CROPS, "candidate plate crops"))


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
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
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
