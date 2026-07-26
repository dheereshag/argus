import os
import sys
import time
import json
import argparse
import pprint
import io

from PIL import Image, ImageOps
import numpy as np

from app.services import PlateRecognizerFactory
from app.schemas.plate import ProviderEnum
from app.services.yolo_filter import filter_vehicle_and_occupancy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy, get_paddle_ocr_engine
from app.services.constants import INDIAN_PLATE_REGEX
from app.core.config import settings

TRAIN_DIR  = "data/images/train"
VAL_DIR    = "data/images/val"
DEBUG_DIR  = "eval_debug_crops"   # saved crops land here

pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


# ── helpers ──────────────────────────────────────────────────────────────────

def save_crop(img_pil: Image.Image, label: str, img_name: str) -> str:
    """Save a PIL crop to DEBUG_DIR and return the path."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    stem = os.path.splitext(img_name)[0]
    out_path = os.path.join(DEBUG_DIR, f"{stem}__{label}.jpg")
    img_pil.save(out_path)
    return out_path


def ocr_with_debug(img_bytes: bytes, vehicle_box, img_name: str, debug: bool = False):
    """
    Run PaddleOCR exactly as the strategy does (box crop → bottom-50% → full).
    When debug=True, save every crop that was tried.
    Returns (plates, crops_saved).
    """
    engine_strategy = PaddleOCRStrategy()
    paddle = get_paddle_ocr_engine()
    crops_saved = []

    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")

    def run_ocr(crop_pil: Image.Image, label: str):
        arr = np.array(crop_pil)
        results = engine_strategy._extract_plates_from_image_array(arr)
        if debug:
            path = save_crop(crop_pil, label, img_name)
            crops_saved.append({"crop": label, "path": path, "plates_found": results})
        return results

    # Priority 1: vehicle box crop
    if vehicle_box is not None and len(vehicle_box) == 4:
        x1, y1, x2, y2 = vehicle_box
        w, h = pil_img.size
        cb = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        if cb[2] > cb[0] and cb[3] > cb[1]:
            plates = run_ocr(pil_img.crop(cb), "box_crop")
            if plates:
                return plates, crops_saved

    # Priority 2: bottom 50%
    width, height = pil_img.size
    crop_top = int(height * 0.50)
    plates = run_ocr(pil_img.crop((0, crop_top, width, height)), "bottom50")
    if plates:
        return plates, crops_saved

    # Priority 3: full image
    plates = run_ocr(pil_img, "full_image")
    return plates, crops_saved


def get_all_vehicle_boxes(img_bytes: bytes) -> list:
    """Re-run YOLO to get all 4-wheeler bounding boxes."""
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


# ── core evaluator ───────────────────────────────────────────────────────────

def evaluate_dataset(
    split: str,
    image_dir: str,
    providers: list,
    multi_vehicle_ok: bool = False,
    debug_image: str = None,         # filename to save debug crops for, e.g. "09.jpg"
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

        t0     = time.time()
        is_debug = (debug_image is not None and img_name == debug_image)

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

        # ── vehicle boxes for OCR ────────────────────────────────────────────
        if is_eligible:
            vehicle_boxes = [primary_box] if primary_box else []
        elif multi_vehicle_ok and vehicle_count > 1:
            vehicle_boxes = get_all_vehicle_boxes(img_bytes)
            vehicle_type  = f"{vehicle_type} (x{vehicle_count})"
        else:
            vehicle_boxes = []

        # ── OCR across providers ─────────────────────────────────────────────
        status = "NO_PLATE_DETECTED"

        if vehicle_count > 0 or (not yolo_res.get("human_detected") and vehicle_count == 0):

            if len(vehicle_boxes) > 1:
                # Multi-vehicle: produce one detection entry per box
                detections = []
                for box_idx, box in enumerate(vehicle_boxes):
                    box_plate, box_state, box_provider = "N/A", "N/A", "N/A"
                    for provider in providers:
                        try:
                            if provider == ProviderEnum.PADDLEOCR:
                                plates, crops = ocr_with_debug(
                                    img_bytes, box, img_name, debug=is_debug
                                )
                                if is_debug and crops:
                                    print(f"  [DEBUG crops for box {box_idx}]")
                                    for c in crops:
                                        pp.pprint(c)
                            else:
                                engine  = PlateRecognizerFactory.get_recognizer(provider.value)
                                plates  = engine.recognize(img_path)
                                crops   = []

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
                continue   # skip the single-result path below

            else:
                # Single vehicle (or no box): standard waterfall
                plate_num, state_val, used_provider = "N/A", "N/A", "N/A"
                for provider in providers:
                    try:
                        if provider == ProviderEnum.PADDLEOCR:
                            box = vehicle_boxes[0] if vehicle_boxes else None
                            plates, crops = ocr_with_debug(
                                img_bytes, box, img_name, debug=is_debug
                            )
                            if is_debug and crops:
                                print(f"  [DEBUG crops for {img_name}]")
                                for c in crops:
                                    pp.pprint(c)
                        else:
                            engine = PlateRecognizerFactory.get_recognizer(provider.value)
                            plates = engine.recognize(img_path)

                        if plates:
                            plate_num    = plates[0].get("plate", "N/A")
                            state_val    = plates[0].get("state", "N/A")
                            used_provider = provider.value
                            status       = "SUCCESS"
                            break
                    except Exception:
                        continue

        elif yolo_res.get("human_detected"):
            status       = "REJECTED_HUMAN"
            plate_num    = "N/A"
            state_val    = "N/A"
            used_provider = "N/A"
        else:
            plate_num    = "N/A"
            state_val    = "N/A"
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
    parser.add_argument("--debug", metavar="FILENAME",
                        help="Save debug crops for a specific image, e.g. --debug 09.jpg")
    args = parser.parse_args()

    providers = (
        [ProviderEnum.PADDLEOCR, ProviderEnum.NVIDIA, ProviderEnum.PLATERECOGNIZER]
        if args.waterfall else [ProviderEnum.PADDLEOCR]
    )

    all_rows = []
    all_rows.extend(evaluate_dataset(
        "train", TRAIN_DIR, providers,
        multi_vehicle_ok=True, debug_image=args.debug
    ))
    all_rows.extend(evaluate_dataset(
        "val", VAL_DIR, providers,
        multi_vehicle_ok=False, debug_image=args.debug
    ))

    out_json = "eval_report.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nFull report saved to: {out_json}")
