import os
import base64
import re
import requests
from typing import List, Dict, Any
from decouple import config

from anpr.base import BasePlateRecognizer
from anpr.constants import INDIAN_PLATE_REGEX, STATE_CODES


class PlateRecognizerStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using Plate Recognizer API.
    """

    def __init__(self, token: str = None, regions: List[str] = None):
        self.api_token = token or config("PLATE_RECOGNIZER_TOKEN", default="")
        self.regions = regions or ["in"]
        self.api_url = "https://api.platerecognizer.com/v1/plate-reader/"

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        if not self.api_token:
            raise ValueError("PLATE_RECOGNIZER_TOKEN is missing in environment variables.")

        with open(image_path, "rb") as fp:
            response = requests.post(
                self.api_url,
                data=dict(regions=self.regions),
                files=dict(upload=fp),
                headers={"Authorization": f"Token {self.api_token}"}
            )

        if response.status_code not in (200, 201):
            print(f"[PlateRecognizerStrategy] Error {response.status_code}: {response.text}")
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
            elif res.get("plate"):
                output.append(self.parse_plate_info(res.get("plate")))

        return output


class NvidiaVisionStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using NVIDIA Llama-3.2-11b-Vision model.
    """

    def __init__(self, api_key: str = None, invoke_url: str = None, model_name: str = None):
        self.api_key = api_key or config("NVIDIA_API_KEY", default="")
        self.invoke_url = invoke_url or config("NVIDIA_INVOKE_URL", default="https://integrate.api.nvidia.com/v1/chat/completions")
        self.model_name = model_name or "meta/llama-3.2-11b-vision-instruct"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing in environment variables.")

        base64_image = self._encode_image(image_path)
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

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

                # Fallback clean text
                clean_str = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
                if clean_str:
                    return [self.parse_plate_info(clean_str)]
            else:
                print(f"[NvidiaVisionStrategy] Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[NvidiaVisionStrategy] Exception: {e}")

        return []
