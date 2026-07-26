import os
from app.services import PlateRecognizerFactory

TESTS_DIR = "tests"

def test_models():
    images = [f for f in os.listdir(TESTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    # Instantiate strategies using Factory
    platerecognizer_service = PlateRecognizerFactory.get_recognizer("platerecognizer")
    nvidia_service = PlateRecognizerFactory.get_recognizer("nvidia")

    for img_name in images:
        img_path = os.path.join(TESTS_DIR, img_name)
        print(f"\n{'='*60}\nTesting image: {img_path}\n{'='*60}")
        
        pr_result = platerecognizer_service.recognize(img_path)
        print("[Plate Recognizer Strategy]:", pr_result)

        nv_result = nvidia_service.recognize(img_path)
        print("[NVIDIA Vision Strategy]:    ", nv_result)

if __name__ == "__main__":
    test_models()
