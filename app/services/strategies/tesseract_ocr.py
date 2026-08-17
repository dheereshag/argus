import re
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.contracts import bounded
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX
from app.services.image_processing import load_rgb


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


class TesseractStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using Tesseract OCR (via pytesseract) for local license plate text extraction.
    Inherits multi-tier ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, bottom_crop_ratio: float = 0.50):
        self.bottom_crop_ratio = bottom_crop_ratio

    def _extract_plates_from_image_array(self, img_pil: Image.Image) -> List[Dict[str, Any]]:
        try:
            import pytesseract  # noqa: PLC0415
            config = f"--psm {settings.TESSERACT_PSM}"
            raw_text = pytesseract.image_to_string(img_pil, config=config, lang=settings.TESSERACT_LANG)
        except Exception as e:
            logger.error(f"[tesseract] Failed to perform OCR: {e}")
            return []

        raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not raw_lines:
            return []

        lines = bounded(raw_lines, settings.MAX_OCR_LINES, "OCR text lines")

        clean_lines: List[str] = []
        for line in lines:
            cand_clean = re.sub(r'[^A-Za-z0-9]', '', line).upper()
            if cand_clean:
                clean_lines.append(cand_clean)

        raw_text_summary = " ".join(clean_lines) if clean_lines else "N/A"

        def _norm(s: str) -> str:
            return "WB" + s[2:] if s.startswith("W8") else s

        detected_plates: List[Dict[str, Any]] = []

        # 1. Inspect individual text lines
        for cand_raw in clean_lines:
            cand_clean = _norm(cand_raw)
            match = INDIAN_PLATE_REGEX.fullmatch(cand_clean)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info:
                    info["raw_text"] = raw_text_summary
                    if info not in detected_plates:
                        detected_plates.append(info)

        # 2. Inspect adjacent and near-adjacent text line pairs (for 2-line Indian plates)
        if not detected_plates and clean_lines:
            n = len(clean_lines)
            for i in range(n):
                # Try adjacent pair (i, i+1)
                if i + 1 < n:
                    pair_str = _norm(clean_lines[i] + clean_lines[i + 1])
                    match = INDIAN_PLATE_REGEX.fullmatch(pair_str)
                    if match:
                        info = self.parse_plate_info(match.group(0))
                        if info:
                            info["raw_text"] = raw_text_summary
                            if info not in detected_plates:
                                detected_plates.append(info)

                # Try near-adjacent pair (i, i+2)
                if i + 2 < n and not detected_plates:
                    pair_str = _norm(clean_lines[i] + clean_lines[i + 2])
                    match = INDIAN_PLATE_REGEX.fullmatch(pair_str)
                    if match:
                        info = self.parse_plate_info(match.group(0))
                        if info:
                            info["raw_text"] = raw_text_summary
                            if info not in detected_plates:
                                detected_plates.append(info)

        # 3. Fallback: If no valid Indian plate matched, but OCR detected text lines
        if not detected_plates and clean_lines:
            detected_plates.append({
                "plate": "N/A",
                "state": "N/A",
                "raw_text": raw_text_summary
            })

        return detected_plates

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """Process a single image crop or full image with Tesseract OCR."""
        pil_img = load_rgb(image_input)
        return self._extract_plates_from_image_array(pil_img)
