import json
import os
import time
from typing import Any, Dict, List, Optional

from app.core.contracts import bounded
from app.services.factory import PlateRecognizerFactory
from app.services.image_processing import decode_and_downscale
from app.services.yolo_filter import filter_vehicle_and_occupancy

TESTS_DIR = "tests"
OUTPUT_JSON = "comparison_report.json"
OUTPUT_MD = "comparison_report.md"

# 40 RPM limit = 1.5s per request. We use 1.8s delay to safely stay below limits.
NVIDIA_MIN_INTERVAL_SEC = 1.8
last_nvidia_call_time = 0.0


def rate_limited_call(fn, *args, **kwargs):
    global last_nvidia_call_time
    now = time.time()
    elapsed = now - last_nvidia_call_time
    if elapsed < NVIDIA_MIN_INTERVAL_SEC:
        sleep_dur = NVIDIA_MIN_INTERVAL_SEC - elapsed
        time.sleep(sleep_dur)

    res = fn(*args, **kwargs)
    last_nvidia_call_time = time.time()
    return res


def extract_plate_string(results: List[Dict[str, Any]]) -> str:
    if not results or not isinstance(results, list):
        return "N/A"
    for r in results:
        if isinstance(r, dict) and r.get("plate") and r.get("plate") != "N/A":
            return str(r.get("plate")).strip().upper()
    return "N/A"


def main():
    print("=" * 80)
    print("ARGUS MULTI-STRATEGY ANPR BENCHMARK & DISCREPANCY DETECTOR")
    print("=" * 80)

    # Initialize strategies
    docling_engine = PlateRecognizerFactory.get_recognizer("docling")
    nvidia_engine = PlateRecognizerFactory.get_recognizer("nvidia")
    pr_engine = PlateRecognizerFactory.get_recognizer("platerecognizer")

    image_files = sorted(
        [
            f for f in os.listdir(TESTS_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )

    print(f"Total test images: {len(image_files)}")
    print("Evaluating strategies: docling (RapidOCR), nvidia (Vision LLM), platerecognizer (Cloud API)")
    print("-" * 80)

    report_entries = []
    discrepancy_count = 0
    all_match_count = 0

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(TESTS_DIR, filename)
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        img_bytes = decode_and_downscale(raw_bytes)

        # YOLO Pre-screening
        t0 = time.time()
        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        t_yolo = round((time.time() - t0) * 1000, 1)

        v_type = yolo_res.get("vehicle_type") or "None"
        v_box = yolo_res.get("vehicle_box")
        v_boxes = yolo_res.get("vehicle_boxes")

        # 1. Docling (RapidOCR)
        t_doc_start = time.time()
        doc_res = docling_engine.recognize(img_bytes, vehicle_box=v_box, vehicle_boxes=v_boxes)
        t_doc = round((time.time() - t_doc_start) * 1000, 1)
        doc_plate = extract_plate_string(doc_res)

        # 2. NVIDIA Vision
        t_nv_start = time.time()
        try:
            nv_res = rate_limited_call(
                nvidia_engine.recognize,
                img_bytes,
                filename=filename,
                vehicle_box=v_box,
                vehicle_boxes=v_boxes
            )
            nv_plate = extract_plate_string(nv_res)
        except Exception as e:
            nv_plate = f"ERR: {e}"
        t_nv = round((time.time() - t_nv_start) * 1000, 1)

        # 3. Plate Recognizer
        t_pr_start = time.time()
        try:
            pr_res = pr_engine.recognize(
                img_bytes,
                filename=filename,
                vehicle_box=v_box,
                vehicle_boxes=v_boxes
            )
            pr_plate = extract_plate_string(pr_res)
        except Exception as e:
            pr_plate = f"ERR: {e}"
        t_pr = round((time.time() - t_pr_start) * 1000, 1)

        # Comparison logic
        valid_plates = [doc_plate, nv_plate, pr_plate]
        all_match = (doc_plate == nv_plate == pr_plate) and (doc_plate != "N/A" and not doc_plate.startswith("ERR"))
        has_discrepancy = not (doc_plate == nv_plate == pr_plate)

        if all_match:
            all_match_count += 1
            status_indicator = "[MATCH]"
        elif has_discrepancy:
            discrepancy_count += 1
            status_indicator = "[DISCREPANCY]"

        entry = {
            "index": idx,
            "filename": filename,
            "vehicle_type": v_type,
            "docling_plate": doc_plate,
            "docling_latency_ms": t_doc,
            "nvidia_plate": nv_plate,
            "nvidia_latency_ms": t_nv,
            "platerecognizer_plate": pr_plate,
            "platerecognizer_latency_ms": t_pr,
            "all_match": all_match,
            "has_discrepancy": has_discrepancy
        }
        report_entries.append(entry)

        print(
            f"[{idx:2d}/{len(image_files)}] {filename:<8} {status_indicator:<14} | "
            f"Docling: {doc_plate:<12} ({t_doc:>5.1f}ms) | "
            f"NVIDIA: {nv_plate:<12} ({t_nv:>6.1f}ms) | "
            f"PR: {pr_plate:<12} ({t_pr:>6.1f}ms)"
        )

    # Save JSON Report
    full_report = {
        "total_images": len(image_files),
        "all_match_count": all_match_count,
        "discrepancy_count": discrepancy_count,
        "match_percentage": f"{(all_match_count / len(image_files)) * 100:.1f}%",
        "results": report_entries
    }

    with open(OUTPUT_JSON, "w") as fp:
        json.dump(full_report, fp, indent=2)

    # Save Markdown Report
    md_lines = [
        "# ANPR Strategy Comparison Report",
        "",
        f"- **Total Images Evaluated**: {len(image_files)}",
        f"- **Full 3-Way Match**: {all_match_count} ({(all_match_count / len(image_files)) * 100:.1f}%)",
        f"- **Discrepancies / Variations**: {discrepancy_count}",
        "",
        "| # | Image | Vehicle | Docling (RapidOCR) | NVIDIA Vision | Plate Recognizer | All Match? | Discrepancy? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |"
    ]

    for e in report_entries:
        match_str = "✅" if e["all_match"] else "❌"
        disc_str = "⚠️" if e["has_discrepancy"] else "—"
        md_lines.append(
            f"| {e['index']} | `{e['filename']}` | {e['vehicle_type']} | **`{e['docling_plate']}`** ({e['docling_latency_ms']}ms) | `{e['nvidia_plate']}` ({e['nvidia_latency_ms']}ms) | `{e['platerecognizer_plate']}` ({e['platerecognizer_latency_ms']}ms) | {match_str} | {disc_str} |"
        )

    with open(OUTPUT_MD, "w") as fp:
        fp.write("\n".join(md_lines))

    print("=" * 80)
    print(f"Report saved to: {OUTPUT_JSON} and {OUTPUT_MD}")
    print(f"Summary: {all_match_count} full matches, {discrepancy_count} discrepancies out of {len(image_files)} images.")
    print("=" * 80)


if __name__ == "__main__":
    main()
