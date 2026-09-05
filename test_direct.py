"""
Direct Model Benchmark & Testing Utility.

Iterates over sample test images in the `tests/` directory and benchmarks:
  1. YOLO v11 Vehicle Pre-screening & Occupancy Filtering latency.
  2. RapidOCR Text Recognition latency and plate candidate output.

Usage:
    uv run python test_direct.py
"""

import argparse
import os
import time

from app.services.detector import filter_vehicle_and_occupancy
from app.services.image_processing import decode_and_downscale
from app.services.ocr import PlateRecognizer

TESTS_DIR = "tests"


def test_models() -> None:
    """
    Run sequential inference across all test images and report component execution times.

    Discovers all JPEG and PNG images in TESTS_DIR, loads and downscales each image,
    and reports execution times for YOLO detection and RapidOCR separately.
    """
    parser = argparse.ArgumentParser(description="ANPR Direct Testing CLI")
    parser.parse_args()

    # Discover and sort test image files
    image_paths = [
        os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_paths.sort()

    engine = PlateRecognizer()

    for img_path in image_paths:
        print(f"\n{'=' * 60}\nTesting image: {img_path}\n{'=' * 60}")

        # Read binary file and apply safety downscaling
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        img_bytes = decode_and_downscale(raw_bytes)

        # ----------------------------------------------------------------------
        # Benchmark 1: YOLO v11 Pre-screening & Occupancy Check
        # ----------------------------------------------------------------------
        t_yolo_start = time.time()
        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 2)
        print(
            f"[YOLO v11 Prescreening] ({t_yolo:>7.2f} ms): vehicle={yolo_res.vehicle_type}, count={yolo_res.vehicle_count}"
        )

        # ----------------------------------------------------------------------
        # Benchmark 2: RapidOCR Direct Recognition
        # ----------------------------------------------------------------------
        t0 = time.time()
        result = engine.recognize(img_bytes)
        t_exec = round((time.time() - t0) * 1000, 2)
        print(f"[RapidOCR             ] ({t_exec:>7.2f} ms): {result}")


if __name__ == "__main__":
    test_models()
