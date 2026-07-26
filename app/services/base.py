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
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Template method executing recognition with 5-tier ROI fallback:
          1. Tier 1: Bottom 1/3 ROI (~33.3% bottom of vehicle crop or frame)
          2. Tier 2: Bottom 1/2 ROI (50% bottom of vehicle crop or frame)
          3. Tier 3: Bottom 2/3 ROI (~66.7% bottom of vehicle crop or frame)
          4. Tier 4: Full vehicle crop (if vehicle_box is present)
          5. Tier 5: Full original image frame fallback
        """
        raw_text_fallbacks = []

        # Tier 1: Bottom 1/3 ROI (bumper level)
        roi_1_3 = crop_image_roi(image_input, vehicle_box, bottom_crop_ratio=1.0 / 3.0, bottom_roi_only=True)
        res1 = self._recognize_single_image(roi_1_3, filename=filename)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res1):
            return res1
        if res1:
            raw_text_fallbacks.append(res1)

        # Tier 2: Bottom 1/2 ROI (50% bottom)
        roi_1_2 = crop_image_roi(image_input, vehicle_box, bottom_crop_ratio=0.50, bottom_roi_only=True)
        res2 = self._recognize_single_image(roi_1_2, filename=filename)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res2):
            return res2
        if res2:
            raw_text_fallbacks.append(res2)

        # Tier 3: Bottom 2/3 ROI (66.7% bottom)
        roi_2_3 = crop_image_roi(image_input, vehicle_box, bottom_crop_ratio=2.0 / 3.0, bottom_roi_only=True)
        res3 = self._recognize_single_image(roi_2_3, filename=filename)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res3):
            return res3
        if res3:
            raw_text_fallbacks.append(res3)

        # Tier 4: Full vehicle crop (if vehicle_box is present)
        if vehicle_box:
            full_crop_bytes = crop_image_roi(image_input, vehicle_box, bottom_roi_only=False)
            res4 = self._recognize_single_image(full_crop_bytes, filename=filename)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res4):
                return res4
            if res4:
                raw_text_fallbacks.append(res4)

        # Tier 5: Full original image frame fallback
        res5 = self._recognize_single_image(image_input, filename=filename)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res5):
            return res5
        if res5:
            raw_text_fallbacks.append(res5)

        # If no valid plate matched in any tier, return the first raw text fallback
        return raw_text_fallbacks[0] if raw_text_fallbacks else []

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
