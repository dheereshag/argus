import argparse
import os
import time

from app.schemas.plate import ProviderEnum
from app.services import PlateRecognizerFactory
from app.services.yolo_filter import filter_vehicle_and_occupancy

TESTS_DIR = "tests"

def parse_args():
    available_providers = [p.value for p in PlateRecognizerFactory.list_providers()]
    parser = argparse.ArgumentParser(description="ANPR Strategy Performance & Direct Testing CLI")
    parser.add_argument(
        "strategies",
        nargs="*",
        choices=available_providers,
        default=available_providers,
        help=f"Recognition strategies to test ({', '.join(available_providers)}). Defaults to all."
    )
    return parser.parse_args()

def test_models():
    args = parse_args()

    image_paths = [
        os.path.join(TESTS_DIR, f)
        for f in os.listdir(TESTS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    image_paths.sort()

    selected_strategies = args.strategies
    print(f"Testing strategies: {', '.join(selected_strategies)}")

    strategy_engines = {}
    for st_name in selected_strategies:
        try:
            strategy_engines[st_name] = PlateRecognizerFactory.get_recognizer(st_name)
        except Exception as e:
            print(f"Error instantiating strategy '{st_name}': {e}")

    for img_path in image_paths:
        print(f"\n{'='*60}\nTesting image: {img_path}\n{'='*60}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t_yolo_start = time.time()
        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 2)
        print(f"[YOLO v11 Prescreening] ({t_yolo:>7.2f} ms): vehicle={yolo_res['vehicle_type']}, box={yolo_res.get('vehicle_box')}")

        for st_name, engine in strategy_engines.items():
            t0 = time.time()
            if st_name == ProviderEnum.PADDLEOCR.value:
                result = engine.recognize(img_bytes, vehicle_box=yolo_res.get("vehicle_box"))
            else:
                result = engine.recognize(img_path)
            t_exec = round((time.time() - t0) * 1000, 2)
            print(f"[{st_name.upper():<20}] ({t_exec:>7.2f} ms): {result}")

if __name__ == "__main__":
    test_models()
