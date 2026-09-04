import argparse
import json
import sys

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.services.detector import VehicleDetector
from app.services.ocr import PlateRecognizer
from app.services.pipeline import recognize_plate_image


def main():
    parser = argparse.ArgumentParser(description=f"{settings.PROJECT_NAME} CLI")
    parser.add_argument("image", help="Path to image file for plate recognition")

    args = parser.parse_args()

    # Warm YOLO model & check OCR engines
    try:
        VehicleDetector.get_model()
        logger.info("YOLO v11 model loaded successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError) as e:
        logger.warning(f"Warning loading YOLO model: {e}")

    PlateRecognizer.check_engine()

    try:
        response = recognize_plate_image(args.image)
        logger.info(json.dumps(response.model_dump(), indent=2))
    except (ANPRServiceError, ContractViolation, ValueError, OSError, RuntimeError) as e:
        logger.error(f"Error processing image '{args.image}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
