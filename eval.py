import os
import sys
import time
import json
import argparse

from app.services import PlateRecognizerFactory
from app.schemas.plate import ProviderEnum
from app.services.yolo_filter import filter_vehicle_and_occupancy

TRAIN_DIR = "data/images/train"
VAL_DIR   = "data/images/val"


def get_all_vehicle_boxes(img_bytes: bytes) -> list:
    """Re-run YOLO to extract all 4-wheeler bounding boxes from an image."""
    from app.services.yolo_filter import get_yolo_model, FOUR_WHEELER_CLASS_NAMES
    from PIL import Image, ImageOps
    import io
    from app.core.config import settings

    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    model = get_yolo_model()
    raw = model(pil_img, verbose=False)[0]
    boxes = []
    if raw.boxes is not None and len(raw.boxes) > 0:
        cls_ids = raw.boxes.cls.cpu().numpy()
        confs   = raw.boxes.conf.cpu().numpy()
        xyxy    = raw.boxes.xyxy.cpu().numpy() if hasattr(raw.boxes, "xyxy") else None
        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            if int(cls_id) in FOUR_WHEELER_CLASS_NAMES and conf >= settings.VEHICLE_CONF_THRESH:
                if xyxy is not None and idx < len(xyxy):
                    boxes.append(tuple(map(int, xyxy[idx])))
    return boxes


def evaluate_dataset(split: str, image_dir: str, providers: list, multi_vehicle_ok: bool = False) -> list:
    if not os.path.exists(image_dir):
        print(f"  [SKIP] Directory '{image_dir}' does not exist.")
        return []

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    provider_names = " -> ".join(p.value for p in providers)
    mode_str = f"WATERFALL ({provider_names})" if len(providers) > 1 else f"{providers[0].value.upper()} ONLY"

    print(f"\n{'='*95}")
    print(f"Evaluating {split.upper()} split: {len(image_files)} images  |  Mode: {mode_str}")
    print(f"{'='*95}")

    header_fmt = "| {:<10} | {:<24} | {:<12} | {:<14} | {:<16} | {:<16} | {:<10} |"
    print(header_fmt.format("Filename", "Status", "Vehicle", "Engine Used", "Plate Detected", "State", "Time (ms)"))
    print("|" + "-"*12 + "|" + "-"*26 + "|" + "-"*14 + "|" + "-"*16 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*12 + "|")

    report_rows = []
    total_start = time.time()

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()

        try:
            yolo_res = filter_vehicle_and_occupancy(img_bytes)
        except Exception as e:
            t_exec = round((time.time() - t0) * 1000, 2)
            row = {
                "split": split, "filename": img_name, "status": "ERROR",
                "vehicle_type": "N/A", "provider": "N/A",
                "plate": "N/A", "state": str(e)[:60], "exec_time_ms": t_exec,
            }
            report_rows.append(row)
            print(header_fmt.format(img_name, "ERROR (corrupt image)", "N/A", "N/A", "N/A", "N/A", t_exec))
            continue

        is_eligible   = yolo_res.get("is_eligible", False)
        vehicle_count = yolo_res.get("vehicle_count", 0)
        vehicle_type  = yolo_res.get("vehicle_type") or "unfiltered"
        primary_box   = yolo_res.get("vehicle_box")

        # Determine vehicle boxes for OCR
        if is_eligible:
            vehicle_boxes = [primary_box] if primary_box else []
        elif multi_vehicle_ok and vehicle_count > 1:
            vehicle_boxes = get_all_vehicle_boxes(img_bytes)
            vehicle_type  = f"{vehicle_type} (x{vehicle_count})"
        else:
            vehicle_boxes = []

        plate_num, state, used_provider = "N/A", "N/A", "N/A"
        status = "NO_PLATE_DETECTED"

        if vehicle_count > 0 or (not yolo_res.get("human_detected") and vehicle_count == 0):
            for provider in providers:
                try:
                    engine = PlateRecognizerFactory.get_recognizer(provider.value)
                    if provider == ProviderEnum.PADDLEOCR:
                        # Try each vehicle box, then full image fallback
                        ocr_results = None
                        for box in vehicle_boxes:
                            ocr_results = engine.recognize(img_bytes, vehicle_box=box)
                            if ocr_results:
                                break
                        if not ocr_results:
                            ocr_results = engine.recognize(img_bytes, vehicle_box=None)
                    else:
                        ocr_results = engine.recognize(img_path)

                    if ocr_results:
                        plate_num    = ocr_results[0].get("plate", "N/A")
                        state        = ocr_results[0].get("state", "N/A")
                        used_provider = provider.value
                        status        = "SUCCESS"
                        break
                except Exception as e:
                    continue
        elif yolo_res.get("human_detected"):
            status = "REJECTED_HUMAN"

        t_exec = round((time.time() - t0) * 1000, 2)

        row = {
            "split": split, "filename": img_name, "status": status,
            "vehicle_type": vehicle_type, "provider": used_provider,
            "plate": plate_num, "state": state, "exec_time_ms": t_exec,
        }
        report_rows.append(row)
        print(header_fmt.format(img_name, status, vehicle_type, used_provider, plate_num, state, t_exec))

    total_duration = round(time.time() - total_start, 2)
    success_count  = sum(1 for r in report_rows if r["status"] == "SUCCESS")
    print(f"{'='*95}")
    print(f"{split.upper()} Summary: {success_count}/{len(report_rows)} SUCCESS  |  Total: {total_duration}s")

    return report_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ANPR on train and val datasets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--paddleocr", action="store_true",
        help="Use PaddleOCR only (no fallback)"
    )
    group.add_argument(
        "--waterfall", action="store_true",
        help="Use full waterfall: PaddleOCR -> NVIDIA -> PlateRecognizer"
    )
    args = parser.parse_args()

    if args.waterfall:
        providers = [ProviderEnum.PADDLEOCR, ProviderEnum.NVIDIA, ProviderEnum.PLATERECOGNIZER]
    else:
        # Default is paddleocr-only (same as --paddleocr)
        providers = [ProviderEnum.PADDLEOCR]

    all_rows = []
    all_rows.extend(evaluate_dataset("train", TRAIN_DIR, providers, multi_vehicle_ok=True))
    all_rows.extend(evaluate_dataset("val",   VAL_DIR,   providers, multi_vehicle_ok=False))

    out_json = "eval_report.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nFull report saved to: {out_json}")
