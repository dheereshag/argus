from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import settings
from app.core.contracts import bounded, require
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import box_area, crop_image_roi, warp_perspective_crop


class BasePlateRecognizer(ABC):
    """
    Abstract Base Class for License Plate Recognition Strategies.
    Implements a unified Template Method with multi-tier ROI and perspective warping fallback.
    """

    @abstractmethod
    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """
        Subclasses must implement this to perform OCR/Vision extraction on a single image.
        Failure to do so raises TypeError at instantiation time, not silently at runtime.
        """

    def recognize(  # noqa: C901
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg",
        vehicle_box: Optional[Tuple[int, int, int, int]] = None,
        vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Template method executing recognition with 5-tier ROI & perspective warping fallback:
          1. Tier 1: Bottom 1/3 ROI (~33.3% bottom of vehicle crop or frame) & perspective warped fallback
          2. Tier 2: Bottom 1/2 ROI (50% bottom of vehicle crop or frame) & perspective warped fallback
          3. Tier 3: Bottom 2/3 ROI (~66.7% bottom of vehicle crop or frame)
          4. Tier 4: Full vehicle crop (if vehicle box is present)
          5. Tier 5: Full original image frame fallback

        If multiple vehicle_boxes are detected, iterates through each vehicle crop until a valid plate is found.

        BOUNDED WORK (NASA rule 2). The tier ladder is a fixed 5, but the box
        list was previously whatever YOLO returned, and cost multiplies:

            len(boxes) x 5 tiers x 2 (plain + warped) x N providers

        A yard with parked vehicles in frame could therefore drive tens of OCR
        calls for one weighing, which is where the 27.4 s outlier in
        eval_report.json comes from. Boxes are area-sorted here and capped at
        MAX_VEHICLE_BOXES, so total work per request has a stated ceiling.
        """
        require(image_input is not None, "recognize() called with no image")

        raw_text_fallbacks = []

        # Determine list of vehicle boxes to inspect.
        candidate_boxes = vehicle_boxes if vehicle_boxes else ([vehicle_box] if vehicle_box else [])

        # Sort largest-first so the cap keeps the vehicle most likely to be the
        # one on the platform. The caller already sorts, but this function does
        # not get to assume that (rule 7 — validate what the caller hands you).
        ordered = sorted(
            (b for b in candidate_boxes if b),
            key=box_area,
            reverse=True,
        )
        boxes_to_check: List[Optional[Tuple[int, int, int, int]]] = list(
            bounded(ordered, settings.MAX_VEHICLE_BOXES, "vehicle boxes")
        )
        if not boxes_to_check:
            boxes_to_check = [None]

        # Helper to try standard crop first, then perspective-warped crop
        def _try_crop(crop_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
            res = self._recognize_single_image(crop_bytes, filename=filename)
            if any(r.get("plate") and r.get("plate") != "N/A" for r in res):
                return res
            if res:
                raw_text_fallbacks.append(res)

            # Try perspective warped & de-skewed crop
            warped_bytes = warp_perspective_crop(crop_bytes)
            if warped_bytes != crop_bytes:
                res_warped = self._recognize_single_image(warped_bytes, filename=filename)
                if any(r.get("plate") and r.get("plate") != "N/A" for r in res_warped):
                    return res_warped
                if res_warped:
                    raw_text_fallbacks.append(res_warped)
            return None

        # Fallback ROI specifications: (bottom_crop_ratio, bottom_roi_only)
        # Tier 1: Bottom 1/3 ROI, Tier 2: Bottom 1/2 ROI, Tier 3: Bottom 2/3 ROI, Tier 4: Full vehicle crop
        tier_specs = ((1.0 / 3.0, True), (0.50, True), (2.0 / 3.0, True), (1.0, False))

        # Iterate across all vehicle bounding boxes through ROI tiers
        for box in boxes_to_check:
            for ratio, bottom_only in tier_specs:
                if not bottom_only and not box:
                    continue
                roi_bytes = crop_image_roi(image_input, box, bottom_crop_ratio=ratio, bottom_roi_only=bottom_only)
                res = _try_crop(roi_bytes)
                if res:
                    return res

        # Tier 5: Full original image frame fallback
        res5 = self._recognize_single_image(image_input, filename=filename)
        if any(r.get("plate") and r.get("plate") != "N/A" for r in res5):
            return res5
        if res5:
            raw_text_fallbacks.append(res5)

        return raw_text_fallbacks[0] if raw_text_fallbacks else []

    def parse_plate_info(self, raw_plate: str) -> Optional[Dict[str, Any]]:
        """
        Validate a candidate string against the Indian plate regex and map the
        state code to a full state name.

        The candidate must match the plate pattern in its ENTIRETY (fullmatch).
        A substring match is not sufficient: text lifted off a vehicle surface
        routinely contains plate-shaped substrings that are not plates
        ("GOODYEAR2024" -> "ODYEAR2024", "ASHOKLEYLAND2820" -> "LAND2820").

        Returns None when the candidate is not a well-formed plate. Callers must
        treat None as "no plate found" and continue their fallback chain.
        """
        if not raw_plate:
            return None

        clean_cand = raw_plate.strip().upper()
        if len(clean_cand) >= 2 and clean_cand[:2] == "W8":  
            clean_cand = "WB" + clean_cand[2:]

        match = INDIAN_PLATE_REGEX.fullmatch(clean_cand)
        if not match:
            return None

        matched_plate = match.group(0).replace(" ", "").replace(".", "").replace("-", "").upper()
        state_code = match.group(1) or match.group(6)
        state_name = STATE_CODES.get(state_code.upper(), "Unknown State") if state_code else "Unknown State"
        return {
            "plate": matched_plate,
            "state": state_name
        }
