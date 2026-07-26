from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import crop_image_roi


class BasePlateRecognizer(ABC):
    """
    Abstract Base Class for License Plate Recognition Strategies.
    Implements a unified Template Method for 3-tier fallback execution:
      1. Bottom half ROI of vehicle crop (bumper/plate level)
      2. Full vehicle crop
      3. Full image frame fallback
    """

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """
        Subclasses implement this method to perform OCR/Vision extraction on a single image.
        """
        return []

    def recognize(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg",
        vehicle_box: Optional[Tuple[int, int, int, int]] = None,
        bottom_crop_ratio: float = 0.50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Template method executing recognition with 3-tier ROI fallback.
        """
        # Tier 1: Bottom half ROI of vehicle crop (or bottom half of full frame)
        roi_bytes = crop_image_roi(image_input, vehicle_box, bottom_crop_ratio=bottom_crop_ratio, bottom_roi_only=True)
        results = self._recognize_single_image(roi_bytes, filename=filename)
        if results:
            return results

        # Tier 2: Full vehicle crop (if vehicle_box is present)
        if vehicle_box:
            full_crop_bytes = crop_image_roi(image_input, vehicle_box, bottom_crop_ratio=bottom_crop_ratio, bottom_roi_only=False)
            results = self._recognize_single_image(full_crop_bytes, filename=filename)
            if results:
                return results

        # Tier 3: Full original image frame fallback
        return self._recognize_single_image(image_input, filename=filename)

    def parse_plate_info(self, raw_plate: str) -> Optional[Dict[str, Any]]:
        """
        Helper utility to validate raw plate string against Indian plate regex
        and map the state code to full state name.
        """
        if not raw_plate:
            return None

        clean_cand = raw_plate.strip().upper()
        if len(clean_cand) >= 2 and clean_cand[:2] == "W8":
            clean_cand = "WB" + clean_cand[2:]

        match = INDIAN_PLATE_REGEX.search(clean_cand)

        if match:
            matched_plate = match.group(0).replace(" ", "").replace(".", "").replace("-", "").upper()
            state_code = match.group(1) or match.group(6)
            state_name = STATE_CODES.get(state_code.upper(), "Unknown State") if state_code else "Unknown State"
            return {
                "plate": matched_plate,
                "state": state_name
            }

        state_code = clean_cand[:2]
        return {
            "plate": clean_cand,
            "state": STATE_CODES.get(state_code, "Unknown State")
        }
