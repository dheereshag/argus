import os
import base64
import re
import requests
from typing import List, Dict, Any, Union, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX

class NvidiaVisionStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using NVIDIA Vision API (Llama 3.2 11B Vision / Nemotron).
    Supports multi-key authentication across LLAMA_API_KEY and NEMOTRON_API_KEY.
    """

    def __init__(self, api_key: str = None, invoke_url: str = None, model_name: str = None):
        self.api_key = api_key
        self.invoke_url = invoke_url or settings.NVIDIA_INVOKE_URL
        self.model_name = model_name or "meta/llama-3.2-11b-vision-instruct"

    def _get_api_keys(self) -> List[str]:
        keys = []
        if self.api_key:
            keys.append(self.api_key)
        if settings.NEMOTRON_API_KEY and settings.NEMOTRON_API_KEY not in keys:
            keys.append(settings.NEMOTRON_API_KEY)
        if settings.LLAMA_API_KEY and settings.LLAMA_API_KEY not in keys:
            keys.append(settings.LLAMA_API_KEY)
        if settings.NVIDIA_API_KEY and settings.NVIDIA_API_KEY not in keys:
            keys.append(settings.NVIDIA_API_KEY)
        return keys

    def _get_base64_and_mime(self, image_input: Union[str, bytes], filename: str) -> Tuple[str, str]:
        if isinstance(image_input, bytes):
            base64_str = base64.b64encode(image_input).decode("utf-8")
            ext = os.path.splitext(filename)[1].lower().replace(".", "")
        else:
            with open(image_input, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode("utf-8")
            ext = os.path.splitext(image_input)[1].lower().replace(".", "")

        mime_type = "image/jpeg" if ext in ("jpg", "jpeg", "") else f"image/{ext}"
        return base64_str, mime_type

    def recognize(self, image_input: Union[str, bytes], filename: str = "image.jpg") -> List[Dict[str, Any]]:
        keys_to_try = self._get_api_keys()
        if not keys_to_try:
            raise ValueError("No NVIDIA API keys (LLAMA_API_KEY / NEMOTRON_API_KEY) configured in settings/env.")

        base64_image, mime_type = self._get_base64_and_mime(image_input, filename)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify and extract the Indian vehicle license plate number from this image. Return ONLY the license plate alphanumeric string (e.g. RJ09GA0165 or MH01AB1234)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "model": self.model_name,
            "max_tokens": 128,
            "temperature": 0.1,
            "stream": False
        }

        for key in keys_to_try:
            headers = {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
            try:
                response = requests.post(self.invoke_url, headers=headers, json=payload)
                if response.status_code == 200:
                    res_json = response.json()
                    raw_text = res_json['choices'][0]['message']['content'].strip()

                    matches = list(INDIAN_PLATE_REGEX.finditer(raw_text))
                    output = []
                    for match in matches:
                        info = self.parse_plate_info(match.group(0))
                        if info:
                            output.append(info)

                    if output:
                        return output
                else:
                    logger.error(f"[NvidiaVisionStrategy] Error {response.status_code} with key '{key[:12]}...': {response.text}")
            except Exception as e:
                logger.error(f"[NvidiaVisionStrategy] Exception with key '{key[:12]}...': {e}")

        return []
