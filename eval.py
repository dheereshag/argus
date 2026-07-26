import os
import sys
import time
import json
import argparse
import pprint
import io

from PIL import Image, ImageOps

from app.services import PlateRecognizerFactory
from app.schemas.plate import ProviderEnum
from app.services.yolo_filter import filter_vehicle_and_occupancy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy
from app.core.config import settings

TRAIN_DIR = "data/images/train"
VAL_DIR   = "data/images/val"

pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_all_vehicle_boxes(img_bytes: bytes) -> list:
    """Re-run YOLO to get all 4-wheeler bounding boxes from an image."""
    from app.services.yolo_filter import get_yolo_model, FOUR_WHEELER_CLASS_NAMES

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


def run_paddle_ocr(img_bytes: bytes, vehicle_box) -> list:
    """Run PaddleOCR on a single box via the strategy."""
    engine = PaddleOCRStrategy()
    results = engine.recognize(img_bytes, vehicle_box=vehicle_box)
    if not results and vehicle_box is not None:
        results = engine.recognize(img_bytes, vehicle_box=None)
    return results or []


# ── core evaluator ────────────────────────────────────────────────────────────

def evaluate_dataset(
    split: str,
    image_dir: str,
    providers: list,
    multi_vehicle_ok: bool = False,
) -> list:
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
            pp.pprint(row)
            continue

        is_eligible   = yolo_res.get("is_eligible", False)
        vehicle_count = yolo_res.get("vehicle_count", 0)
        vehicle_type  = yolo_res.get("vehicle_type") or "unfiltered"
        primary_box   = yolo_res.get("vehicle_box")

        # ── vehicle boxes for OCR ─────────────────────────────────────────────
        if is_eligible:
            vehicle_boxes = [primary_box] if primary_box else []
        elif multi_vehicle_ok and vehicle_count > 1:
            vehicle_boxes = get_all_vehicle_boxes(img_bytes)
            vehicle_type  = f"{vehicle_type} (x{vehicle_count})"
        else:
            vehicle_boxes = []

        # ── OCR ───────────────────────────────────────────────────────────────
        status = "NO_PLATE_DETECTED"

        if vehicle_count > 0 or (not yolo_res.get("human_detected") and vehicle_count == 0):

            if len(vehicle_boxes) > 1:
                # Multi-vehicle: one detection dict per box
                detections = []
                for box_idx, box in enumerate(vehicle_boxes):
                    box_plate, box_state, box_provider = "N/A", "N/A", "N/A"
                    for provider in providers:
                        try:
                            if provider == ProviderEnum.PADDLEOCR:
                                plates = run_paddle_ocr(img_bytes, box)
                            else:
                                engine = PlateRecognizerFactory.get_recognizer(provider.value)
                                plates = engine.recognize(img_path)

                            if plates:
                                box_plate    = plates[0].get("plate", "N/A")
                                box_state    = plates[0].get("state", "N/A")
                                box_provider = provider.value
                                break
                        except Exception:
                            continue

                    detections.append({
                        "box_index": box_idx,
                        "box": box,
                        "plate": box_plate,
                        "state": box_state,
                        "provider": box_provider,
                    })

                any_success = any(d["plate"] != "N/A" for d in detections)
                status = "SUCCESS" if any_success else "NO_PLATE_DETECTED"

                row = {
                    "split": split,
                    "filename": img_name,
                    "status": status,
                    "vehicle_type": vehicle_type,
                    "detections": detections,
                    "exec_time_ms": round((time.time() - t0) * 1000, 2),
                }
                report_rows.append(row)
                pp.pprint(row)
                continue

            else:
                # Single vehicle (or no box): waterfall
                plate_num, state_val, used_provider = "N/A", "N/A", "N/A"
                for provider in providers:
                    try:
                        if provider == ProviderEnum.PADDLEOCR:
                            box    = vehicle_boxes[0] if vehicle_boxes else None
                            plates = run_paddle_ocr(img_bytes, box)
                        else:
                            engine = PlateRecognizerFactory.get_recognizer(provider.value)
                            plates = engine.recognize(img_path)

                        if plates:
                            plate_num     = plates[0].get("plate", "N/A")
                            state_val     = plates[0].get("state", "N/A")
                            used_provider = provider.value
                            status        = "SUCCESS"
                            break
                    except Exception:
                        continue

        elif yolo_res.get("human_detected"):
            status        = "REJECTED_HUMAN"
            plate_num     = "N/A"
            state_val     = "N/A"
            used_provider = "N/A"
        else:
            plate_num     = "N/A"
            state_val     = "N/A"
            used_provider = "N/A"

        t_exec = round((time.time() - t0) * 1000, 2)
        row = {
            "split": split,
            "filename": img_name,
            "status": status,
            "vehicle_type": vehicle_type,
            "provider": used_provider,
            "plate": plate_num,
            "state": state_val,
            "exec_time_ms": t_exec,
        }
        report_rows.append(row)
        pp.pprint(row)

    total_duration = round(time.time() - total_start, 2)
    success_count  = sum(1 for r in report_rows if r["status"] == "SUCCESS")
    print(f"{'='*95}")
    print(f"{split.upper()} Summary: {success_count}/{len(report_rows)} SUCCESS  |  Total: {total_duration}s")

    return report_rows


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ANPR on train and val datasets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paddleocr", action="store_true", help="PaddleOCR only")
    group.add_argument("--waterfall", action="store_true",
                       help="Full waterfall: PaddleOCR -> NVIDIA -> PlateRecognizer")
    args = parser.parse_args()

    providers = (
        [ProviderEnum.PADDLEOCR, ProviderEnum.NVIDIA, ProviderEnum.PLATERECOGNIZER]
        if args.waterfall else [ProviderEnum.PADDLEOCR]
    )

    all_rows = []
    all_rows.extend(evaluate_dataset("train", TRAIN_DIR, providers, multi_vehicle_ok=True))
    all_rows.extend(evaluate_dataset("val",   VAL_DIR,   providers, multi_vehicle_ok=False))

    out_json = "eval_report.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nFull report saved to: {out_json}")
