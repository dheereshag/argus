"""End-to-End Two-Stage ANPR Pipeline Orchestrator."""

import time

from app.schemas import RecognitionResponse
from app.services.image_processing import decode_image
from app.services.pipeline.helpers import _resolve_bytes
from app.services.pipeline.response import _build_response
from app.services.pipeline.stages import _run_stage2_ocr


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """Execute the Two-Stage ANPR Pipeline (YOLO26 detection + RapidOCR) on an input image."""
    start_time = time.time()
    resolved_filename = filename or (image_input if isinstance(image_input, str) else "image.jpg")
    prepared_img = decode_image(_resolve_bytes(image_input))

    import app.services.pipeline as pl

    detection = pl.VehicleDetector().detect(prepared_img)
    plate_results = _run_stage2_ocr(detection, prepared_img, resolved_filename)
    return _build_response(detection, resolved_filename, start_time, results=plate_results)

