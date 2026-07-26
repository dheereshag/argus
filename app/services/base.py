from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES

class BasePlateRecognizer(ABC):
    """
    Abstract Strategy Interface for License Plate Recognition Models.
    """

    @abstractmethod
    def recognize(self, image_input: Union[str, bytes], filename: str = "image.jpg") -> List[Dict[str, Any]]:
        """
        Recognize license plates from an image file path or raw bytes.
        Returns a list of dicts containing plate details:
        [{"plate": "RJ09GA0165", "state": "Rajasthan"}]
        """
        pass

    def parse_plate_info(self, raw_plate: str) -> Dict[str, Any]:
        """
        Helper utility to validate raw plate string against Indian plate regex
        and map the state code to full state name.
        """
        if not raw_plate:
            return None

        clean_cand = raw_plate.strip().upper()
        match = INDIAN_PLATE_REGEX.search(clean_cand)

        if match:
            matched_plate = match.group(0).replace(" ", "").replace(".", "").replace("-", "").upper()
            state_code = match.group(1) or match.group(6)
            state_name = STATE_CODES.get(state_code.upper(), "Unknown State") if state_code else "Unknown State"
            return {
                "plate": matched_plate,
                "state": state_name
            }
        
        # Fallback if regex search missed
        state_code = clean_cand[:2]
        return {
            "plate": clean_cand,
            "state": STATE_CODES.get(state_code, "Unknown State")
        }
