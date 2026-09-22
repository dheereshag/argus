"""Direct Model Benchmark & Testing Utility (`uv run python test_direct.py`)."""

import os
import time

from app.services.detector import VehicleDetector
from app.services.image_processing import decode_and_downscale
from app.services.ocr import PlateRecognizer

TESTS_DIR = "tests"


def test_models() -> None:
    """Run sequential inference across all test images and report execution times."""
    image_paths = sorted(
        os.path.join(TESTS_DIR, f)
        for f in os.listdir(TESTS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    detector = VehicleDetector()
    engine = PlateRecognizer()

    for img_path in image_paths:
        print(f"\n{'=' * 60}\nTesting image: {img_path}\n{'=' * 60}")
        with open(img_path, "rb") as f:
            prep_img = decode_and_downscale(f.read())

        t_yolo_start = time.time()
        yolo_res = detector.detect(prep_img)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 2)
        v_types = [v.vehicle_type for v in yolo_res.vehicles]
        print(f"[YOLO26 Prescreening] ({t_yolo:>7.2f} ms): vehicles={v_types}")

        t0 = time.time()
        result = engine.recognize(prep_img)
        t_exec = round((time.time() - t0) * 1000, 2)
        print(f"[RapidOCR             ] ({t_exec:>7.2f} ms): {result}")


if __name__ == "__main__":
    test_models()
