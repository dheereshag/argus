import os
from typing import Any, Dict, List, Union

import requests
from requests.exceptions import RequestException

from app.core.config import settings
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX


class PlateRecognizerStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using Plate Recognizer Cloud API.
    Inherits 3-tier vehicle crop & bottom ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, token: str | None = None, regions: List[str] | None = None):
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

        # Check file size < 3.5MB before making API call
        MAX_FILE_SIZE_BYTES = int(3.5 * 1024 * 1024)  # 3.5 MB limit

        if isinstance(image_input, bytes):
            img_bytes = image_input
            file_size = len(img_bytes)
        else:
            file_size = os.path.getsize(image_input)
            with open(image_input, "rb") as fp:
                img_bytes = fp.read()

        if file_size >= MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"[PlateRecognizerStrategy] Skipping API call: file size {file_size / (1024 * 1024):.2f}MB "
                f"exceeds 3.5MB limit ({file_size} bytes >= {MAX_FILE_SIZE_BYTES} bytes)."
            )
            return []

        files = dict(upload=(filename, img_bytes, "image/jpeg"))
        try:
            response = requests.post(
                self.api_url,
                data=dict(regions=self.regions),
                files=files,
                headers={"Authorization": f"Token {self.api_token}"},
                timeout=(settings.HTTP_CONNECT_TIMEOUT, settings.HTTP_READ_TIMEOUT),
            )
        except RequestException as exc:
            logger.error(f"[PlateRecognizerStrategy] Network error: {exc}")
            return []

        if response.status_code not in (200, 201):
            logger.error(f"[PlateRecognizerStrategy] Error {response.status_code}: {response.text}")
            return []

        res_data = response.json()
        results = res_data.get("results", [])
        output = []

        all_raw_cands = []
        for res in results:
            candidates = [res.get("plate", "")] + [c.get("plate", "") for c in res.get("candidates", [])]
            valid_info = None

            for cand in candidates:
                if not cand:
                    continue
                all_raw_cands.append(cand.upper())
                info = self.parse_plate_info(cand)
                if info and INDIAN_PLATE_REGEX.fullmatch(info["plate"]):
                    info["raw_text"] = cand.upper()
                    valid_info = info
                    break

            if valid_info:
                output.append(valid_info)

        if output:
            return output
        elif all_raw_cands:
            return [{"plate": "N/A", "state": "N/A", "raw_text": " ".join(all_raw_cands)}]

        return []
