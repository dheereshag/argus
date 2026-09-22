"""Direct Model Benchmark & Testing Utility (`uv run python test_direct.py`)."""

import os

from app.schemas import RecognitionResponse
from app.services.pipeline import recognize_plate_image

TESTS_DIR = "tests"


def test_models() -> list[RecognitionResponse]:
    """Run full ANPR pipeline across all test images and output API response payloads."""
    image_paths = sorted(
        os.path.join(TESTS_DIR, f)
        for f in os.listdir(TESTS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    responses: list[RecognitionResponse] = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\n{'=' * 60}\nTesting image: {img_path}\n{'=' * 60}")
        resp = recognize_plate_image(img_path, filename=filename)
        print(resp.model_dump_json(indent=2))
        responses.append(resp)

    return responses


if __name__ == "__main__":
    test_models()
