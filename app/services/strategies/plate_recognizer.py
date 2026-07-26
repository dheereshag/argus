import requests
from typing import List, Dict, Any, Union
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX


class PlateRecognizerStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using Plate Recognizer Cloud API.
    Inherits 3-tier vehicle crop & bottom ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, token: str = None, regions: List[str] = None):
        self.api_token = token or settings.PLATE_RECOGNIZER_TOKEN
        self.regions = regions or ["in"]
        self.api_url = "https://api.platerecognizer.com/v1/plate-reader/"

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """Process a single image crop or full image with Plate Recognizer API."""
        if not self.api_token:
            raise ValueError("PLATE_RECOGNIZER_TOKEN is missing in settings/env.")

        if isinstance(image_input, bytes):
            files = dict(upload=(filename, image_input, "image/jpeg"))
            response = requests.post(
                self.api_url,
                data=dict(regions=self.regions),
                files=files,
                headers={"Authorization": f"Token {self.api_token}"}
            )
        else:
            with open(image_input, "rb") as fp:
                response = requests.post(
                    self.api_url,
                    data=dict(regions=self.regions),
                    files=dict(upload=fp),
                    headers={"Authorization": f"Token {self.api_token}"}
                )

        if response.status_code not in (200, 201):
            logger.error(f"[PlateRecognizerStrategy] Error {response.status_code}: {response.text}")
            return []

        res_data = response.json()
        results = res_data.get("results", [])
        output = []

        for res in results:
            candidates = [res.get("plate", "")] + [c.get("plate", "") for c in res.get("candidates", [])]
            valid_info = None

            for cand in candidates:
                if not cand:
                    continue
                info = self.parse_plate_info(cand)
                if info and INDIAN_PLATE_REGEX.fullmatch(info["plate"]):
                    valid_info = info
                    break

            if valid_info:
                output.append(valid_info)

        return output
