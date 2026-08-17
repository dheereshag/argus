import argparse
import io
import json
import os
import pprint
import sys
import time

from PIL import Image, ImageOps
from tabulate import tabulate

from app.core.config import settings
from app.eval.metrics import (
    compare_to_baseline,
    evaluate,
    format_report,
    format_worst_offenders,
    load_labels,
)
from app.schemas.plate import ProviderEnum
from app.services import PlateRecognizerFactory
from app.services.image_processing import save_debug_images
from app.services.yolo_filter import filter_vehicle_and_occupancy

TRAIN_DIR = "data/images/train"
VAL_DIR   = "data/images/val"
LABELS_PATH = "data/labels.csv"
BASELINE_PATH = "eval_baseline.json"

pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_all_vehicle_boxes(img_bytes: bytes) -> list:
    """Re-run YOLO to get all 4-wheeler bounding boxes from an image."""
    from app.services.yolo_filter import FOUR_WHEELER_CLASS_NAMES, get_yolo_model  # noqa: PLC0415

    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    model = get_yolo_model()
    raw = model(pil_img, verbose=False)[0]
    boxes = []
    if raw.boxes is not None and len(raw.boxes) > 0:
        cls_ids = raw.boxes.cls.cpu().numpy()
        confs   = raw.boxes.conf.cpu().numpy()
        xyxy    = raw.boxes.xyxy.cpu().numpy() if hasattr(raw.boxes, "xyxy") else None
        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs, strict=False)):
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


def report_accuracy(rows: list, labels_path: str, baseline_path: str,
                    write_baseline: bool = False) -> int:
    """
    Score the run against ground truth and print the accuracy report.

    Returns a process exit code: 0 if fine, 1 if the run regressed against the
    saved baseline. That exit code is what CI (issue #11) gates on.

    The status counts printed by print_summary_table are an EXTRACTION RATE —
    how often the pipeline emitted something plate-shaped. They are retained
    because they are useful for debugging the pipeline, but they are not
    accuracy and must not be quoted as such.
    """
    try:
        labels = load_labels(labels_path)
    except FileNotFoundError as exc:
        print(f"\n{'=' * 68}")
        print("ACCURACY: NOT MEASURED")
        print("=" * 68)
        print(f"  {exc}")
        print("=" * 68)
        return 0

    metrics = evaluate(rows, labels)
    print()
    print(format_report(metrics))

    offenders = format_worst_offenders(metrics)
    if offenders:
        print()
        print(offenders)

    exit_code = 0
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        verdict = compare_to_baseline(metrics, baseline)
        print()
        print("=" * 68)
        print(f"REGRESSION CHECK vs {baseline_path}")
        print("=" * 68)
        if verdict.improvements:
            for item in verdict.improvements:
                print(f"  IMPROVED  {item}")
        if verdict.failures:
            for item in verdict.failures:
                print(f"  REGRESSED {item}")
            print("\n  This run is worse than the baseline. Do not merge.")
            exit_code = 1
        elif not verdict.improvements:
            print("  No material change.")
        print("=" * 68)
    else:
        print(f"\n  No baseline at '{baseline_path}'. "
              f"Run with --write-baseline to create one.")

    if write_baseline:
        with open(baseline_path, "w") as fh:
            json.dump(metrics.to_dict(), fh, indent=2)
        print(f"\n  Baseline written to {baseline_path}")

    return exit_code


def print_summary_table(rows: list) -> None:
    """
    Pretty print status counts grouped by split.

    NOTE: this is an extraction rate, not accuracy. See report_accuracy.
    """
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
        summary_data.append([s, *stats])

    total_stats = get_stats(rows)
    summary_data.append(["TOTAL", *total_stats])

    print("\nEVALUATION SUMMARY:")
    print(tabulate(summary_data, headers=headers, tablefmt="rounded_grid"))


# ── core evaluator ────────────────────────────────────────────────────────────

def evaluate_dataset(  # noqa: C901, PLR0917, PLR0912, PLR0915
    split: str,
    image_dir: str,
    providers: list,
    multi_vehicle_ok: bool = False,
    target_files: list | None = None,
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

        vehicle_count = yolo_res.get("vehicle_count", 0)
        vehicle_type  = yolo_res.get("vehicle_type") or "unfiltered"

        # ── vehicle boxes for OCR ─────────────────────────────────────────────
        vehicle_boxes = get_all_vehicle_boxes(img_bytes) if vehicle_count > 0 else []

        if save_crops:
            save_debug_images(img_bytes, img_name, vehicle_boxes, vehicle_type, output_dir=output_dir)

        # ── OCR ───────────────────────────────────────────────────────────────
        status = "NO_PLATE_DETECTED"
        plate_num, state_val, used_provider, raw_text_val = "N/A", "N/A", "N/A", "N/A"

        if vehicle_count > 0 or (not yolo_res.get("human_detected") and vehicle_count == 0):
            for provider in providers:
                try:
                    engine = PlateRecognizerFactory.get_recognizer(provider.value)
                    plates = engine.recognize(img_bytes, filename=img_name, vehicle_boxes=vehicle_boxes)

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
            status = "REJECTED_HUMAN"

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
    group.add_argument("--tesseract", action="store_true", help="Tesseract OCR only")
    group.add_argument("--waterfall", action="store_true",
                       help="Full waterfall: Tesseract -> NVIDIA -> PlateRecognizer")

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
    parser.add_argument(
        "--labels",
        default=LABELS_PATH,
        help=f"Ground-truth CSV to score against (default: {LABELS_PATH}). See LABELLING.md"
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_PATH,
        help=f"Baseline metrics to compare against (default: {BASELINE_PATH})"
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite the baseline with this run's metrics. Use deliberately."
    )
    args = parser.parse_args()

    target_files = []
    if args.images:
        target_files.extend(args.images)
    if args.files:
        target_files.extend(args.files)

    providers = (
        [ProviderEnum.TESSERACT, ProviderEnum.NVIDIA, ProviderEnum.PLATERECOGNIZER]
        if args.waterfall else [ProviderEnum.TESSERACT]
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

    exit_code = report_accuracy(
        all_rows,
        labels_path=args.labels,
        baseline_path=args.baseline,
        write_baseline=args.write_baseline,
    )
    sys.exit(exit_code)

