import re
import threading
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.core.config import settings
from app.core.contracts import bounded, ensure
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX
from app.services.image_processing import load_rgb

# Lazily-built OCR engine, guarded by a lock (NASA rule 6).
#
# Same race as the YOLO singleton: the check-then-set was unsynchronised, and
# the request handler now runs in FastAPI's threadpool. PaddleOCR construction
# is heavy — two concurrent first-requests building two engines is a genuine
# memory spike on a Pi. main.py also warms this during lifespan.
_PADDLE_OCR_INSTANCE: Optional[Any] = None
_PADDLE_OCR_LOCK = threading.Lock()


def get_paddle_ocr_engine() -> Any:
    global _PADDLE_OCR_INSTANCE  # noqa: PLW0603
    if _PADDLE_OCR_INSTANCE is None:
        with _PADDLE_OCR_LOCK:
            if _PADDLE_OCR_INSTANCE is None:  # re-check under the lock
                logger.debug(
                    f"Initializing PaddleOCR engine instance "
                    f"(cpu_threads={settings.PADDLE_CPU_THREADS}, "
                    f"angle_cls={settings.PADDLE_USE_ANGLE_CLS})."
                )
                from paddleocr import PaddleOCR  # noqa: PLC0415
                _PADDLE_OCR_INSTANCE = PaddleOCR(
                    use_angle_cls=settings.PADDLE_USE_ANGLE_CLS,
                    lang="en",
                    show_log=False,
                    cpu_threads=settings.PADDLE_CPU_THREADS
                )
    ensure(_PADDLE_OCR_INSTANCE is not None, "PaddleOCR engine failed to initialise")
    return _PADDLE_OCR_INSTANCE


class PaddleOCRStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using PaddleOCR for local license plate detection & recognition.
    Inherits 3-tier vehicle crop & bottom ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, bottom_crop_ratio: float = 0.50):
        self.bottom_crop_ratio = bottom_crop_ratio

    def _extract_plates_from_image_array(self, img_np: np.ndarray) -> List[Dict[str, Any]]:  # noqa: C901, PLR0912
        ocr_engine = get_paddle_ocr_engine()
        ocr_results = ocr_engine.ocr(img_np, cls=settings.PADDLE_USE_ANGLE_CLS)

        if not ocr_results or not ocr_results[0]:
            return []

        # Bounded (rule 2). The pairing pass below is O(n) with two windows, and
        # a frame full of signage or a text-covered tarpaulin yields many lines,
        # none of which are plates. Cap the input rather than the loop body so
        # the bound is stated once.
        lines = bounded(ocr_results[0], settings.MAX_OCR_LINES, "OCR text lines")

        detected_plates: List[Dict[str, Any]] = []
        clean_lines: List[str] = []

        # Collect raw text lines. Each entry is ((box), (text, score)); tolerate
        # a malformed row rather than failing the whole read (rule 7).
        for line in lines:
            try:
                text_str, _score = line[1]
            except (IndexError, TypeError, ValueError):
                logger.warning(f"[paddle] skipping malformed OCR row: {line!r:.80}")
                continue
            cand_clean = re.sub(r'[^A-Za-z0-9]', '', str(text_str)).upper()
            if cand_clean:
                clean_lines.append(cand_clean)

        raw_text_summary = " ".join(clean_lines) if clean_lines else "N/A"

        def _norm(s: str) -> str:
            return "WB" + s[2:] if s.startswith("W8") else s

        # 1. Inspect individual text boxes.
        # fullmatch, not search: an OCR line must BE a plate, not merely contain
        # a plate-shaped substring. See parse_plate_info for why.
        for cand_raw in clean_lines:
            cand_clean = _norm(cand_raw)
            match = INDIAN_PLATE_REGEX.fullmatch(cand_clean)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info:
                    info["raw_text"] = raw_text_summary
                    if info not in detected_plates:
                        detected_plates.append(info)

        # 2. Inspect adjacent and near-adjacent text box pairs (for 2-line Indian plates)
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
        """Process a single image crop or full image with PaddleOCR."""
        return self._extract_plates_from_image_array(np.array(load_rgb(image_input)))
