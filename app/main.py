import argparse
import json
import sys

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.services.pipeline import recognize_plate_image
from app.services.strategies.docling_ocr import check_docling_engine
from app.services.yolo_filter import get_yolo_model


def main():
    parser = argparse.ArgumentParser(description=f"{settings.PROJECT_NAME} CLI")
    parser.add_argument("image", help="Path to image file for plate recognition")

    args = parser.parse_args()

    # Warm YOLO model & check OCR engines
    try:
        get_yolo_model()
        logger.info("YOLO v11 model loaded successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError) as e:
        logger.warning(f"Warning loading YOLO model: {e}")

    check_docling_engine()

    try:
        response = recognize_plate_image(args.image)
        print(json.dumps(response.model_dump(), indent=2))
    except (ANPRServiceError, ContractViolation, ValueError, OSError, RuntimeError) as e:
        logger.error(f"Error processing image '{args.image}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
