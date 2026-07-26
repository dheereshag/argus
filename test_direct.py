import os
import time
from app.services import PlateRecognizerFactory

TESTS_DIR = "tests"

def test_models():
    images = [f for f in os.listdir(TESTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    # Instantiate strategies using Factory
    platerecognizer_service = PlateRecognizerFactory.get_recognizer("platerecognizer")
    nvidia_service = PlateRecognizerFactory.get_recognizer("nvidia")
    paddleocr_service = PlateRecognizerFactory.get_recognizer("paddleocr")

    for img_name in images:
        img_path = os.path.join(TESTS_DIR, img_name)
        print(f"\n{'='*60}\nTesting image: {img_path}\n{'='*60}")
        
        t0 = time.time()
        pr_result = platerecognizer_service.recognize(img_path)
        t_pr = round((time.time() - t0) * 1000, 2)
        print(f"[Plate Recognizer Strategy] ({t_pr} ms):", pr_result)

        t0 = time.time()
        nv_result = nvidia_service.recognize(img_path)
        t_nv = round((time.time() - t0) * 1000, 2)
        print(f"[NVIDIA Vision Strategy]    ({t_nv} ms):", nv_result)

        t0 = time.time()
        paddle_result = paddleocr_service.recognize(img_path)
        t_paddle = round((time.time() - t0) * 1000, 2)
        print(f"[PaddleOCR Strategy]        ({t_paddle} ms):", paddle_result)

if __name__ == "__main__":
    test_models()
