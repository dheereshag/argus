import io
import re
from abc import ABC, abstractmethod
from typing import Any

from app.core.contracts import require
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.image_processing import ImageInput, load_rgb


class BasePlateRecognizer(ABC):
    """
    Abstract Base Class for ANPR Recognition Strategies.
    Provides standard vehicle crop hierarchy, de-skew fallback, and state resolution.
    """

    @abstractmethod
    def _recognize_single_image(self, image_input: str | bytes, filename: str = "image.jpg") -> list[dict[str, Any]]:
        """
        Subclasses implement OCR / VLM plate extraction on a single image buffer.
        Returns a list of dicts with keys: 'plate', 'state', 'raw_text'.
        """

    def recognize(
        self,
        image_input: ImageInput,
        filename: str = "image.jpg",
    ) -> list[dict[str, Any]]:
        """
        Run OCR plate extraction on the full image frame.
        """
        require(image_input is not None, "recognize() called with no image")

        if isinstance(image_input, str):
            filename = filename or image_input
            with open(image_input, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            with io.BytesIO() as buf:
                load_rgb(image_input).save(buf, format="JPEG")
                image_bytes = buf.getvalue()

        return self._recognize_single_image(image_bytes, filename=filename)

    def parse_plate_info(self, raw_plate: str | None) -> dict[str, Any] | None:
        """
        Validate candidate plate string against Indian plate regex and resolve State/UT.
        Strips whitespace and normalizes known state prefix confusions (e.g. W8 -> WB).
        """
        if not raw_plate:
            return None

        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate)).upper()
        if not cleaned:
            return None

        if cleaned.startswith("W8"):
            cleaned = "WB" + cleaned[2:]

        match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
        if not match:
            return None

        matched_plate = cleaned

        # Standard Indian Series State resolution
        if match.group(1):
            state_code = match.group(1).upper()
            state_name = STATE_CODES.get(state_code, "Unknown State")
            return {"plate": matched_plate, "state": state_name}

        # Bharat (BH) Series
        if match.group(5):  # "BH"
            return {"plate": matched_plate, "state": STATE_CODES.get("BH", "Bharat Series (National)")}

        return {"plate": matched_plate, "state": "Unknown State"}
