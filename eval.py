import os
import sys
import time
import json
import argparse
import pprint
import io
from tabulate import tabulate

from PIL import Image, ImageOps, ImageDraw

from app.services import PlateRecognizerFactory
from app.schemas.plate import ProviderEnum
from app.services.yolo_filter import filter_vehicle_and_occupancy
from app.services.image_processing import save_debug_images
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


def print_detailed_table(rows: list) -> None:
    """Pretty print a detailed table of evaluation results using tabulate."""
    if not rows:
        return

    headers = ["Split", "Filename", "Status", "Vehicle Type", "Provider", "Plate", "State", "Raw OCR Text", "Time (ms)"]

    table_data = []
    for r in rows:
        split = str(r.get("split", "")).upper()
        filename = str(r.get("filename", ""))
        status = str(r.get("status", ""))
        vtype = str(r.get("vehicle_type", "N/A"))

        if "detections" in r and isinstance(r["detections"], list):
            plates = [d["plate"] for d in r["detections"] if isinstance(d, dict) and d.get("plate") and d["plate"] != "N/A"]
            states = [d["state"] for d in r["detections"] if isinstance(d, dict) and d.get("state") and d["state"] != "N/A"]
            provs  = [d["provider"] for d in r["detections"] if isinstance(d, dict) and d.get("provider") and d["provider"] != "N/A"]
            raws   = [d["raw_text"] for d in r["detections"] if isinstance(d, dict) and d.get("raw_text") and d["raw_text"] != "N/A"]
            plate    = ", ".join(dict.fromkeys(plates)) if plates else "N/A"
            state    = ", ".join(dict.fromkeys(states)) if states else "N/A"
            provider = ", ".join(dict.fromkeys(provs)) if provs else "N/A"
            raw_text = ", ".join(dict.fromkeys(raws)) if raws else "N/A"
        else:
            plate    = str(r.get("plate", "N/A"))
            state    = str(r.get("state", "N/A"))
            provider = str(r.get("provider", "N/A"))
            raw_text = str(r.get("raw_text", "N/A"))

        t_exec = f"{r.get('exec_time_ms', 0):.2f}"
        table_data.append([split, filename, status, vtype, provider, plate, state, raw_text, t_exec])

    print("\nDETAILED EVALUATION REPORT:")
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))


def print_summary_table(rows: list) -> None:
    """Pretty print a summary table grouped by split using tabulate."""
    if not rows:
        return

    splits = sorted(list(set(str(r.get("split", "")).upper() for r in rows if r.get("split"))))
    headers = ["Split", "Total", "Success", "No Plate", "Rejected", "Error", "Success Rate", "Avg Time (ms)"]

    def get_stats(subset):
        total = len(subset)
        success  = sum(1 for r in subset if r.get("status") == "SUCCESS")
        no_plate = sum(1 for r in subset if r.get("status") == "NO_PLATE_DETECTED")
        rejected = sum(1 for r in subset if r.get("status") == "REJECTED_HUMAN")
        error    = sum(1 for r in subset if r.get("status") == "ERROR")
        rate     = f"{(success / total * 100):.2f}%" if total > 0 else "0.00%"
        avg_time = f"{(sum(r.get('exec_time_ms', 0) for r in subset) / total):.2f}" if total > 0 else "0.00"
        return [total, success, no_plate, rejected, error, rate, avg_time]

    summary_data = []
    for s in splits:
        subset = [r for r in rows if str(r.get("split", "")).upper() == s]
        stats = get_stats(subset)
        summary_data.append([s] + list(stats))

    total_stats = get_stats(rows)
    summary_data.append(["TOTAL"] + list(total_stats))

    print("\nEVALUATION SUMMARY:")
    print(tabulate(summary_data, headers=headers, tablefmt="rounded_grid"))


# ── core evaluator ────────────────────────────────────────────────────────────

def evaluate_dataset(
    split: str,
    image_dir: str,
    providers: list,
    multi_vehicle_ok: bool = False,
    target_files: list = None,
    save_crops: bool = True,
    output_dir: str = "eval_debug_crops",
) -> list:
    if not os.path.exists(image_dir):
        print(f"  [SKIP] Directory '{image_dir}' does not exist.")
        return []

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if target_files:
        normalized_targets = set()
        for tf in target_files:
            bname = os.path.basename(tf).lower()
            normalized_targets.add(bname)
            if not any(bname.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                for ext in [".jpg", ".jpeg", ".png"]:
                    normalized_targets.add(f"{bname}{ext}")

        image_files = [f for f in image_files if f.lower() in normalized_targets]

    if not image_files:
        return []

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
                "plate": "N/A", "state": str(e)[:60], "raw_text": "N/A", "exec_time_ms": t_exec,
            }
            report_rows.append(row)
            pp.pprint(row)
            continue

        is_eligible   = yolo_res.get("is_eligible", False)
        vehicle_count = yolo_res.get("vehicle_count", 0)
        vehicle_type  = yolo_res.get("vehicle_type") or "unfiltered"
        primary_box   = yolo_res.get("vehicle_box")

        # ── vehicle boxes for OCR ─────────────────────────────────────────────
        if primary_box:
            vehicle_boxes = [primary_box]
        elif multi_vehicle_ok and vehicle_count > 1:
            vehicle_boxes = get_all_vehicle_boxes(img_bytes)
            vehicle_type  = f"{vehicle_type} (x{vehicle_count})"
        else:
            vehicle_boxes = []

        if save_crops:
            save_debug_images(img_bytes, img_name, vehicle_boxes, vehicle_type, output_dir=output_dir)

        # ── OCR ───────────────────────────────────────────────────────────────
        status = "NO_PLATE_DETECTED"

        if vehicle_count > 0 or (not yolo_res.get("human_detected") and vehicle_count == 0):

            if len(vehicle_boxes) > 1:
                # Multi-vehicle: one detection dict per box
                detections = []
                for box_idx, box in enumerate(vehicle_boxes):
                    box_plate, box_state, box_provider, box_raw = "N/A", "N/A", "N/A", "N/A"
                    for provider in providers:
                        try:
                            engine = PlateRecognizerFactory.get_recognizer(provider.value)
                            plates = engine.recognize(img_bytes, filename=img_name, vehicle_box=box)

                            if plates:
                                box_raw      = plates[0].get("raw_text", "N/A")
                                box_provider = provider.value
                                p_cand = plates[0].get("plate", "N/A")
                                if p_cand != "N/A":
                                    box_plate = p_cand
                                    box_state = plates[0].get("state", "N/A")
                                    break
                        except Exception:
                            continue

                    detections.append({
                        "box_index": box_idx,
                        "box": box,
                        "plate": box_plate,
                        "state": box_state,
                        "provider": box_provider,
                        "raw_text": box_raw,
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
                plate_num, state_val, used_provider, raw_text_val = "N/A", "N/A", "N/A", "N/A"
                box = vehicle_boxes[0] if vehicle_boxes else None

                for provider in providers:
                    try:
                        engine = PlateRecognizerFactory.get_recognizer(provider.value)
                        plates = engine.recognize(img_bytes, filename=img_name, vehicle_box=box)

                        if plates:
                            used_provider = provider.value
                            raw_text_val  = plates[0].get("raw_text", "N/A")
                            p_cand = plates[0].get("plate", "N/A")
                            if p_cand != "N/A":
                                plate_num = p_cand
                                state_val = plates[0].get("state", "N/A")
                                status    = "SUCCESS"
                                break
                    except Exception:
                        continue

        elif yolo_res.get("human_detected"):
            status        = "REJECTED_HUMAN"
            plate_num     = "N/A"
            state_val     = "N/A"
            used_provider = "N/A"
            raw_text_val  = "N/A"
        else:
            plate_num     = "N/A"
            state_val     = "N/A"
            used_provider = "N/A"
            raw_text_val  = "N/A"

        t_exec = round((time.time() - t0) * 1000, 2)
        row = {
            "split": split,
            "filename": img_name,
            "status": status,
            "vehicle_type": vehicle_type,
            "provider": used_provider,
            "plate": plate_num,
            "state": state_val,
            "raw_text": raw_text_val,
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

    parser.add_argument(
        "--image", "--images", "-i",
        nargs="+",
        dest="images",
        help="Filter evaluation to specific image filename(s) (e.g. 35.jpg or 35)"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional image filename(s) to evaluate (e.g. 35.jpg or 35)"
    )
    parser.add_argument(
        "--output-dir", "--save-dir",
        default="eval_debug_crops",
        help="Directory to save YOLO bounding box images and OCR crops (default: eval_debug_crops)"
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=True,
        help="Save YOLO bounding box annotated image and crops (default: True)"
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_false",
        dest="save_crops",
        help="Disable saving YOLO box annotated images and crops"
    )
    args = parser.parse_args()

    target_files = []
    if args.images:
        target_files.extend(args.images)
    if args.files:
        target_files.extend(args.files)

    providers = (
        [ProviderEnum.PADDLEOCR, ProviderEnum.NVIDIA, ProviderEnum.PLATERECOGNIZER]
        if args.waterfall else [ProviderEnum.PADDLEOCR]
    )

    all_rows = []
    all_rows.extend(evaluate_dataset(
        "train", TRAIN_DIR, providers, multi_vehicle_ok=True,
        target_files=target_files, save_crops=args.save_crops, output_dir=args.output_dir
    ))
    all_rows.extend(evaluate_dataset(
        "val", VAL_DIR, providers, multi_vehicle_ok=False,
        target_files=target_files, save_crops=args.save_crops, output_dir=args.output_dir
    ))

    out_json = "eval_report.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nFull report saved to: {out_json}")

    print_detailed_table(all_rows)
    print_summary_table(all_rows)

