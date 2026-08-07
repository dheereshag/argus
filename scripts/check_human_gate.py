#!/usr/bin/env python3
"""
Spike harness for issue #1 — does the human-detection gate reject legitimate weighings?

`filter_vehicle_and_occupancy` rejects any frame containing a person above
HUMAN_CONF_THRESH (default 0.30), anywhere in the frame, with no spatial
constraint. If drivers stay in the cab during a weighing they are visible
through the windscreen, and this gate rejects the normal case.

If that is happening, it dominates the rejection rate and every accuracy number
measured afterwards is taken on the wrong population. Hence this runs before the
corpus is labelled, not after.

This script does not decide anything. It gives you the rejection rate, a
threshold sweep, and annotated crops so you can look at what is being rejected
and make the call yourself.

Usage
-----
    uv run python scripts/check_human_gate.py data/images/train
    uv run python scripts/check_human_gate.py data/images/train --save-crops
    uv run python scripts/check_human_gate.py data/images/train --sweep 0.2 0.3 0.5 0.7 0.9

Read the output as
-----------------
    "rejected_human" high  -> the gate is firing on your normal case. Options:
                              restrict it to a region outside the cab, raise the
                              threshold, or drop it for the pilot.
    "rejected_human" ~0    -> the gate is fine. Close issue #1 and move on.
"""

import argparse
import io
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from app.services.yolo_filter import (  # noqa: E402
    FOUR_WHEELER_CLASS_NAMES,
    PERSON_CLASS_ID,
    get_yolo_model,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def detections(img_bytes: bytes):
    """Return (person_detections, vehicle_detections) as lists of (conf, box)."""
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    result = get_yolo_model()(pil_img, verbose=False)[0]

    people, vehicles = [], []
    if result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            box = tuple(map(int, xyxy[idx]))
            if int(cls_id) == PERSON_CLASS_ID:
                people.append((float(conf), box))
            elif int(cls_id) in FOUR_WHEELER_CLASS_NAMES:
                vehicles.append((float(conf), box, FOUR_WHEELER_CLASS_NAMES[int(cls_id)]))
    return people, vehicles


def annotate(img_bytes: bytes, people, vehicles, out_path: str) -> None:
    """
    Draw person boxes in red, vehicle boxes in green.

    The point is to let you see WHERE the person is. A person box overlapping the
    windscreen is a driver in the cab, which is a false rejection. A person box
    beside the vehicle may be a legitimate one.
    """
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    draw = ImageDraw.Draw(img)
    for conf, box, vtype in vehicles:
        draw.rectangle(box, outline="green", width=3)
        draw.text((box[0] + 4, box[1] + 4), f"{vtype} {conf:.2f}", fill="green")
    for conf, box in people:
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, max(0, box[1] - 14)), f"person {conf:.2f}", fill="red")
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("image_dir", help="Directory of real site captures")
    parser.add_argument("--save-crops", action="store_true",
                        help="Write annotated images to --out-dir for eyeballing")
    parser.add_argument("--out-dir", default="human_gate_spike",
                        help="Where annotated images go (default: human_gate_spike)")
    parser.add_argument("--sweep", nargs="+", type=float,
                        default=[0.20, 0.30, 0.50, 0.70, 0.90],
                        help="HUMAN_CONF_THRESH values to sweep (default: .2 .3 .5 .7 .9)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"error: '{args.image_dir}' is not a directory")
        return 2

    files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        print(f"error: no images found in '{args.image_dir}'")
        return 2

    if args.save_crops:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning {len(files)} images in {args.image_dir}\n")

    rows = []
    for name in files:
        with open(os.path.join(args.image_dir, name), "rb") as fh:
            img_bytes = fh.read()
        try:
            people, vehicles = detections(img_bytes)
        except Exception as exc:
            print(f"  {name:24} ERROR {exc}")
            continue

        top_person = max((c for c, _ in people), default=0.0)
        rows.append({"filename": name, "top_person_conf": top_person,
                     "n_people": len(people), "n_vehicles": len(vehicles)})

        if args.save_crops and people:
            annotate(img_bytes, people, vehicles,
                     os.path.join(args.out_dir, f"{os.path.splitext(name)[0]}_gate.jpg"))

        flag = "PERSON" if top_person >= 0.30 else ""
        print(f"  {name:24} people={len(people)}  top_conf={top_person:.2f}  "
              f"vehicles={len(vehicles)}  {flag}")

    if not rows:
        print("\nNo images processed.")
        return 1

    total = len(rows)
    print(f"\n{'=' * 62}")
    print("THRESHOLD SWEEP — share of images the human gate would reject")
    print(f"{'=' * 62}")
    print(f"{'HUMAN_CONF_THRESH':>18} {'rejected':>10} {'of total':>10} {'rate':>8}")
    for thresh in sorted(args.sweep):
        rejected = sum(1 for r in rows if r["top_person_conf"] >= thresh)
        print(f"{thresh:>18.2f} {rejected:>10} {total:>10} {rejected / total:>7.1%}")

    current = sum(1 for r in rows if r["top_person_conf"] >= 0.30)
    no_vehicle = sum(1 for r in rows if r["n_vehicles"] == 0)
    multi_vehicle = sum(1 for r in rows if r["n_vehicles"] > 1)

    print(f"\n{'=' * 62}")
    print("AT THE CURRENT SETTING (HUMAN_CONF_THRESH = 0.30)")
    print(f"{'=' * 62}")
    print(f"  rejected: human detected      {current:>4} / {total}  ({current / total:.1%})")
    print(f"  rejected: no 4-wheeler        {no_vehicle:>4} / {total}  ({no_vehicle / total:.1%})")
    print(f"  rejected: multiple vehicles   {multi_vehicle:>4} / {total}  ({multi_vehicle / total:.1%})")

    surviving = sum(1 for r in rows
                    if r["top_person_conf"] < 0.30 and r["n_vehicles"] == 1)
    print(f"  reaching OCR at all           {surviving:>4} / {total}  ({surviving / total:.1%})")

    print(f"\n  person-confidence distribution: "
          f"{dict(sorted(Counter(round(r['top_person_conf'], 1) for r in rows).items()))}")

    print(f"\n{'=' * 62}")
    if current / total > 0.20:
        print("VERDICT INPUT: the gate rejects a large share of these images.")
        print("Look at the annotated crops. If the person boxes sit on the")
        print("windscreen, this is the driver in the cab and the gate is")
        print("rejecting your normal case. Decide before labelling the corpus.")
    else:
        print("VERDICT INPUT: the gate is quiet on this set. Likely safe to keep")
        print("as-is for the pilot. Confirm the set is representative of real")
        print("captures, not clean stock photos.")
    print(f"{'=' * 62}")

    if args.save_crops:
        print(f"\nAnnotated images with person boxes: {args.out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
