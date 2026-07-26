import io
import re
import numpy as np
from typing import List, Dict, Any, Union
from PIL import Image
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX

_PADDLE_OCR_INSTANCE = None

def get_paddle_ocr_engine():
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE is None:
        logger.debug("Initializing PaddleOCR engine instance.")
        from paddleocr import PaddleOCR
        _PADDLE_OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _PADDLE_OCR_INSTANCE

class PaddleOCRStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using PaddleOCR with Smart Bottom ROI Cropping.
    Focuses OCR detection on the lower bumper area where Indian license plates are mounted,
    filtering out background slogans and commercial vehicle text.
    """

    def __init__(self, bottom_crop_ratio: float = 0.50):
        self.bottom_crop_ratio = bottom_crop_ratio

    def _crop_bottom_roi(self, image_input: Union[str, bytes]) -> np.ndarray:
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = Image.open(image_input).convert("RGB")

        width, height = pil_img.size
        # Crop lower section (default bottom 50%)
        crop_top = int(height * (1.0 - self.bottom_crop_ratio))
        cropped_img = pil_img.crop((0, crop_top, width, height))
        return np.array(cropped_img), np.array(pil_img)

    def _extract_plates_from_image_array(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        ocr_engine = get_paddle_ocr_engine()
        ocr_results = ocr_engine.ocr(img_np, cls=True)

        if not ocr_results or not ocr_results[0]:
            return []

        lines = ocr_results[0]
        detected_plates = []
        combined_text_lines = []

        # 1. Inspect individual text boxes
        for line in lines:
            text_str, score = line[1]
            combined_text_lines.append(text_str)

            cand_clean = re.sub(r'[^A-Za-z0-9]', '', text_str).upper()
            match = INDIAN_PLATE_REGEX.search(cand_clean)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info and info not in detected_plates:
                    detected_plates.append(info)

        # 2. Inspect multi-line combinations (e.g. state code on line 1, number on line 2)
        if not detected_plates and combined_text_lines:
            concatenated = "".join(re.sub(r'[^A-Za-z0-9]', '', t).upper() for t in combined_text_lines)
            match = INDIAN_PLATE_REGEX.search(concatenated)
            if match:
                info = self.parse_plate_info(match.group(0))
                if info:
                    detected_plates.append(info)

        return detected_plates

    def recognize(self, image_input: Union[str, bytes], filename: str = "image.jpg") -> List[Dict[str, Any]]:
        cropped_roi_np, full_img_np = self._crop_bottom_roi(image_input)

        # 1. Primary check: Smart Bottom ROI Crop
        plates = self._extract_plates_from_image_array(cropped_roi_np)
        if plates:
            return plates

        # 2. Fallback check: Full Image Frame
        return self._extract_plates_from_image_array(full_img_np)
