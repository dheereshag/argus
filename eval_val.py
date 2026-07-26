import os
import sys
import time
import json
import argparse
from app.services import PlateRecognizerFactory
from app.schemas.plate import ProviderEnum
from app.services.yolo_filter import filter_vehicle_and_occupancy

VAL_DIR = "data/images/val"

def evaluate_val_dataset(use_waterfall: bool = False):
    if not os.path.exists(VAL_DIR):
        print(f"Error: Directory '{VAL_DIR}' does not exist.")
        sys.exit(1)

    image_files = [f for f in os.listdir(VAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    mode_str = "WATERFALL FALLBACK (PaddleOCR -> NVIDIA -> PlateRecognizer)" if use_waterfall else "PADDLEOCR ONLY"
    print(f"Evaluating {len(image_files)} validation images in mode: {mode_str}...")

    report_rows = []
    total_start = time.time()

    providers_waterfall = [
        ProviderEnum.PADDLEOCR,
        ProviderEnum.NVIDIA,
        ProviderEnum.PLATERECOGNIZER
    ] if use_waterfall else [ProviderEnum.PADDLEOCR]

    for img_name in image_files:
        img_path = os.path.join(VAL_DIR, img_name)
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        vehicle_box = yolo_res.get("vehicle_box")
        vehicle_type = yolo_res.get("vehicle_type") or "unfiltered"

        plate_num = "N/A"
        state = "N/A"
        used_provider = "N/A"
        t0 = time.time()

        for provider in providers_waterfall:
            engine = PlateRecognizerFactory.get_recognizer(provider.value)
            if provider == ProviderEnum.PADDLEOCR:
                ocr_results = engine.recognize(img_bytes, vehicle_box=vehicle_box)
                if not ocr_results and vehicle_box is not None:
                    ocr_results = engine.recognize(img_bytes, vehicle_box=None)
            else:
                ocr_results = engine.recognize(img_path)

            if ocr_results:
                plate_num = ocr_results[0].get("plate", "N/A")
                state = ocr_results[0].get("state", "N/A")
                used_provider = provider.value
                break

        t_exec = round((time.time() - t0) * 1000, 2)

        report_rows.append({
            "filename": img_name,
            "status": "SUCCESS" if plate_num != "N/A" else "NO_PLATE_DETECTED",
            "vehicle_type": vehicle_type,
            "provider": used_provider if plate_num != "N/A" else "N/A",
            "plate": plate_num,
            "state": state,
            "exec_time_ms": t_exec
        })

    total_duration = round(time.time() - total_start, 2)

    print("\n" + "="*95)
    print(f"VALIDATION REPORT ({mode_str}) - Total Elapsed: {total_duration}s")
    print("="*95)
    header_fmt = "| {:<10} | {:<18} | {:<12} | {:<14} | {:<16} | {:<16} | {:<10} |"
    print(header_fmt.format("Filename", "Status", "Vehicle", "Engine Used", "Plate Detected", "State", "Time (ms)"))
    print("|" + "-"*12 + "|" + "-"*20 + "|" + "-"*14 + "|" + "-"*16 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*12 + "|")
    
    for row in report_rows:
        print(header_fmt.format(
            row["filename"],
            row["status"],
            row["vehicle_type"],
            row["provider"],
            row["plate"],
            row["state"],
            row["exec_time_ms"]
        ))
    print("="*95)

    out_json = "eval_val_waterfall_report.json" if use_waterfall else "eval_val_paddle_report.json"
    with open(out_json, "w") as f:
        json.dump(report_rows, f, indent=2)
    print(f"\nDetailed report saved to: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterfall", action="store_true", help="Enable automatic fallback waterfall (PaddleOCR -> NVIDIA -> PlateRecognizer)")
    args = parser.parse_args()
    evaluate_val_dataset(use_waterfall=args.waterfall)
