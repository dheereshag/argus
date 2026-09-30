"""Direct Model Benchmark & Testing Utility (`uv run python test_direct.py`)."""

import os
import re
import statistics
import time
from typing import Any

import numpy as np

from app.schemas import RecognitionResponse
from app.services.pipeline import recognize_plate_image

TESTS_DIR = "tests/images"
REPORT_FILE = "benchmark_report.md"


def preload_models() -> None:
    """Preload and warm up Fast-ALPR, YOLO, and RapidOCR models."""
    print("Preloading model weights and warming up inference engines...")
    from app.services.detector import VehicleDetector
    from app.services.ocr import PlateRecognizer
    from app.services.ocr.fast_alpr_engine import check_fast_alpr_engine

    VehicleDetector.get_model()
    PlateRecognizer.check_engine()
    check_fast_alpr_engine()

    import cv2

    _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    recognize_plate_image(buf.tobytes(), filename="warmup.jpg")
    print("Model preloading and warmup complete.\n")


def natural_key(filename: str) -> int:
    """Sort filenames numerically by leading integer."""
    digits = re.findall(r"\d+", filename)
    return int(digits[0]) if digits else 999999


def serialize_images(directory: str = TESTS_DIR) -> list[str]:
    """Ensure images in directory are consecutively numbered 1.jpg .. N.jpg."""
    if not os.path.exists(directory):
        return []
    valid_exts = (".jpg", ".jpeg", ".png")
    filenames = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(valid_exts)],
        key=natural_key,
    )
    if not filenames:
        return []

    expected = [f"{i}.jpg" for i in range(1, len(filenames) + 1)]
    if filenames == expected:
        return [os.path.join(directory, f) for f in filenames]

    print(f"Serializing {len(filenames)} images in '{directory}'...")
    staged: list[tuple[str, str]] = []
    for i, fname in enumerate(filenames, start=1):
        src = os.path.join(directory, fname)
        temp_name = os.path.join(directory, f"_staging_ser_{i}.tmp")
        os.rename(src, temp_name)
        staged.append((temp_name, os.path.join(directory, f"{i}.jpg")))

    final_paths: list[str] = []
    for temp_path, target_path in staged:
        os.rename(temp_path, target_path)
        final_paths.append(target_path)

    print(f"Serialized {len(final_paths)} images (1.jpg to {len(final_paths)}.jpg).\n")
    return final_paths


def _build_summary_markdown(
    stats: dict[str, Any],
    tier_counts: dict[str, int],
    rows: list[dict[str, Any]],
) -> str:
    """Assemble formatted markdown report content."""
    lines = [
        "# Argus ANPR Benchmark Report\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Dataset**: `{TESTS_DIR}` ({stats['total']} images)\n",
        "## 1. Summary Metrics\n",
        "| Metric | Value |",
        "|---|---|",
        f"| **Total Images Processed** | {stats['total']} |",
        f"| **Images with Plate(s)** | {stats['with_plates']} ({stats['detection_rate']:.1f}%) |",
        f"| **Images without Plate** | {stats['zero_plates']} ({stats['zero_rate']:.1f}%) |",
        f"| **Total Plates Detected** | {stats['total_plates']} |",
        f"| **Total Wall Duration** | {stats['wall_time']:.2f} s |",
        f"| **Mean Latency** | {stats['mean_lat']:.2f} ms |",
        f"| **Median (P50) Latency** | {stats['med_lat']:.2f} ms |",
        f"| **Min Latency** | {stats['min_lat']:.2f} ms |",
        f"| **Max Latency** | {stats['max_lat']:.2f} ms |",
        f"| **P95 Latency** | {stats['p95_lat']:.2f} ms |",
        f"| **P99 Latency** | {stats['p99_lat']:.2f} ms |\n",
        "## 2. Latency Tier Breakdown\n",
        "| Tier | Latency Range | Count | Percentage |",
        "|---|---|---|---|",
        f"| **Fast-ALPR Full-Frame** | $\\le 50\\text{{ ms}}$ | {tier_counts['fast']} | {tier_counts['fast'] / stats['total'] * 100:.1f}% |",
        f"| **Crop / Secondary Tier** | $50\\text{{--}}200\\text{{ ms}}$ | {tier_counts['medium']} | {tier_counts['medium'] / stats['total'] * 100:.1f}% |",
        f"| **RapidOCR Fallback Tier** | $> 200\\text{{ ms}}$ | {tier_counts['slow']} | {tier_counts['slow'] / stats['total'] * 100:.1f}% |\n",
        "## 3. Detailed Results per Image\n",
        "| # | Image | Plates Detected | Latency (ms) | Speed Tier |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        plates_str = ", ".join(f"`{p}`" for p in r["plates"]) if r["plates"] else "*None*"
        lines.append(f"| {r['index']} | `{r['filename']}` | {plates_str} | {r['latency']:.2f} | {r['tier']} |")

    return "\n".join(lines) + "\n"


def generate_report(records: list[dict[str, Any]], total_wall: float) -> str:
    """Compute benchmark statistics and write out markdown report."""
    if not records:
        return "No records to report."

    latencies = [r["latency"] for r in records]
    with_plates = sum(1 for r in records if r["plates"])
    total_plates = sum(len(r["plates"]) for r in records)
    total_imgs = len(records)

    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)

    stats = {
        "total": total_imgs,
        "with_plates": with_plates,
        "zero_plates": total_imgs - with_plates,
        "total_plates": total_plates,
        "detection_rate": (with_plates / total_imgs) * 100,
        "zero_rate": ((total_imgs - with_plates) / total_imgs) * 100,
        "wall_time": total_wall,
        "mean_lat": statistics.mean(latencies),
        "med_lat": statistics.median(latencies),
        "min_lat": min(latencies),
        "max_lat": max(latencies),
        "p95_lat": p95,
        "p99_lat": p99,
    }

    tier_counts = {
        "fast": sum(1 for l in latencies if l <= 50.0),
        "medium": sum(1 for l in latencies if 50.0 < l <= 200.0),
        "slow": sum(1 for l in latencies if l > 200.0),
    }

    for r in records:
        lat = r["latency"]
        r["tier"] = "Fast-ALPR (≤50ms)" if lat <= 50.0 else ("Crop (50-200ms)" if lat <= 200.0 else "RapidOCR (>200ms)")

    report_md = _build_summary_markdown(stats, tier_counts, records)
    with open(REPORT_FILE, "w") as f:
        f.write(report_md)

    print(f"\n{'=' * 60}\nBENCHMARK SUMMARY REPORT\n{'=' * 60}")
    print(f"Total Images:     {stats['total']}")
    print(f"Plates Found:     {stats['total_plates']} across {stats['with_plates']} images ({stats['detection_rate']:.1f}% hit rate)")
    print(f"Mean Latency:     {stats['mean_lat']:.2f} ms")
    print(f"Median (P50):     {stats['med_lat']:.2f} ms")
    print(f"95th Percentile:  {stats['p95_lat']:.2f} ms")
    print(f"Min / Max:        {stats['min_lat']:.2f} ms / {stats['max_lat']:.2f} ms")
    print(f"Fast-ALPR Ratio:  {tier_counts['fast']}/{stats['total']} ({tier_counts['fast'] / stats['total'] * 100:.1f}%)")
    print(f"Report saved to:  {REPORT_FILE}\n{'=' * 60}\n")

    return report_md


def test_models() -> list[RecognitionResponse]:
    """Preload models, serialize test images, run ANPR pipeline, and generate benchmark performance report."""
    image_paths = serialize_images(TESTS_DIR)
    if not image_paths:
        print(f"No test images found in '{TESTS_DIR}'.")
        return []

    preload_models()

    responses: list[RecognitionResponse] = []
    records: list[dict[str, Any]] = []

    start_wall = time.time()
    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        resp = recognize_plate_image(img_path, filename=filename)
        responses.append(resp)
        plate_nums = [r.plate for r in resp.results]
        records.append({
            "index": idx,
            "filename": filename,
            "plates": plate_nums,
            "latency": resp.execution_time_ms,
        })
        print(f"[{idx:02d}/{len(image_paths)}] {filename:10} -> {plate_nums or ['None']} ({resp.execution_time_ms:.2f} ms)")

    total_wall = time.time() - start_wall
    generate_report(records, total_wall)

    return responses


if __name__ == "__main__":
    test_models()
