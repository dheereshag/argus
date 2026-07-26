import io
import re
import numpy as np
from typing import List, Dict, Any, Union
from PIL import Image, ImageOps
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX

_PADDLE_OCR_INSTANCE = None

def get_paddle_ocr_engine():
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE is None:
        logger.debug(
            f"Initializing PaddleOCR engine instance "
            f"(cpu_threads={settings.PADDLE_CPU_THREADS}, angle_cls={settings.PADDLE_USE_ANGLE_CLS})."
        )
        from paddleocr import PaddleOCR
        _PADDLE_OCR_INSTANCE = PaddleOCR(
            use_angle_cls=settings.PADDLE_USE_ANGLE_CLS,
            lang="en",
            show_log=False,
            cpu_threads=settings.PADDLE_CPU_THREADS
        )
    return _PADDLE_OCR_INSTANCE


class PaddleOCRStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using PaddleOCR for local license plate detection & recognition.
    Inherits 3-tier vehicle crop & bottom ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, bottom_crop_ratio: float = 0.50):
        self.bottom_crop_ratio = bottom_crop_ratio

    def _extract_plates_from_image_array(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        ocr_engine = get_paddle_ocr_engine()
        ocr_results = ocr_engine.ocr(img_np, cls=settings.PADDLE_USE_ANGLE_CLS)

        if not ocr_results or not ocr_results[0]:
            return []

        lines = ocr_results[0]
        detected_plates = []
        clean_lines = []

        # 1. Inspect individual text boxes
        for line in lines:
            text_str, score = line[1]
            cand_clean = re.sub(r'[^A-Za-z0-9]', '', text_str).upper()
            if cand_clean:
                clean_lines.append(cand_clean)

            match = INDIAN_PLATE_REGEX.search(cand_clean)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info and info not in detected_plates:
                    detected_plates.append(info)

        # 2. Inspect adjacent and near-adjacent text box pairs (for 2-line Indian plates)
        if not detected_plates and clean_lines:
            n = len(clean_lines)
            for i in range(n):
                # Try adjacent pair (i, i+1)
                if i + 1 < n:
                    pair_str = clean_lines[i] + clean_lines[i + 1]
                    match = INDIAN_PLATE_REGEX.search(pair_str)
                    if match:
                        info = self.parse_plate_info(match.group(0))
                        if info and info not in detected_plates:
                            detected_plates.append(info)

                # Try near-adjacent pair (i, i+2)
                if i + 2 < n and not detected_plates:
                    pair_str = clean_lines[i] + clean_lines[i + 2]
                    match = INDIAN_PLATE_REGEX.search(pair_str)
                    if match:
                        info = self.parse_plate_info(match.group(0))
                        if info and info not in detected_plates:
                            detected_plates.append(info)

        # 3. Global fallback concatenation if adjacent pairing didn't find anything
        if not detected_plates and clean_lines:
            concatenated = "".join(clean_lines)
            match = INDIAN_PLATE_REGEX.search(concatenated)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info:
                    detected_plates.append(info)

        return detected_plates

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """Process a single image crop or full image with PaddleOCR."""
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
        else:
            pil_img = Image.open(image_input)

        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        return self._extract_plates_from_image_array(np.array(pil_img))
