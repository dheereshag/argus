import argparse
import os
import time

from app.services.image_processing import decode_and_downscale
from app.services.strategies.docling_ocr import DoclingStrategy
from app.services.yolo_filter import filter_vehicle_and_occupancy

TESTS_DIR = "tests"


def test_models():
    parser = argparse.ArgumentParser(description="ANPR Docling Direct Testing CLI")
    parser.parse_args()

    image_paths = [
        os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_paths.sort()

    engine = DoclingStrategy()

    for img_path in image_paths:
        print(f"\n{'=' * 60}\nTesting image: {img_path}\n{'=' * 60}")

        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        img_bytes = decode_and_downscale(raw_bytes)

        t_yolo_start = time.time()
        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 2)
        print(
            f"[YOLO v11 Prescreening] ({t_yolo:>7.2f} ms): vehicle={yolo_res['vehicle_type']}, count={yolo_res.get('vehicle_count', 1)}"
        )

        t0 = time.time()
        result = engine.recognize(img_bytes)
        t_exec = round((time.time() - t0) * 1000, 2)
        print(f"[DOCLING              ] ({t_exec:>7.2f} ms): {result}")


if __name__ == "__main__":
    test_models()
