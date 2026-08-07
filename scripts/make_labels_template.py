#!/usr/bin/env python3
"""
Generate a ground-truth label template for the eval corpus (issue #2).

Writes one CSV row per image with `true_plate` blank, ready to fill in by hand.
See LABELLING.md for the format and the normalisation rules.

Usage
-----
    # blank template — you type every plate
    uv run python scripts/make_labels_template.py data/images/train -o data/labels.csv

    # pre-filled from a previous eval run — you verify instead of type
    uv run python scripts/make_labels_template.py data/images/train \
        -o data/labels.csv --seed-from eval_report.json

    # multiple directories into one file
    uv run python scripts/make_labels_template.py data/images/train data/images/val \
        -o data/labels.csv

On seeding
----------
Seeding pre-fills `true_plate` with what the pipeline predicted. It turns typing
into checking, which is faster, but it invites rubber-stamping — and a label that
agrees with the model because nobody looked is worse than no label, because it
makes the model's errors permanently invisible.

Every seeded row is therefore marked SEEDED-VERIFY in `notes`. Delete the marker
as you confirm each row. eval.py counts remaining markers and reports them next
to the accuracy figure, so a reviewer can see how much of the ground truth was
actually verified.

This script never overwrites an existing output file. Labels are expensive.
"""

import argparse
import csv
import json
import os
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SEED_MARKER = "SEEDED-VERIFY"


def collect_images(dirs):
    """Return sorted unique basenames across all given directories."""
    found = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"warning: '{d}' is not a directory, skipping", file=sys.stderr)
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(IMAGE_EXTS) and name not in found:
                found[name] = os.path.join(d, name)
    return found


def load_seed(path):
    """
    Map filename -> predicted plate from an eval_report.json.

    Only rows with a real plate are seeded. 'N/A' and missing values are left
    blank so the labeller reads them fresh rather than confirming a non-answer.
    """
    if not os.path.exists(path):
        print(f"error: seed file '{path}' not found", file=sys.stderr)
        raise SystemExit(2)

    with open(path) as fh:
        rows = json.load(fh)

    seed = {}
    for row in rows:
        name = row.get("filename")
        plate = (row.get("plate") or "").strip()
        if name and plate and plate.upper() != "N/A":
            seed[name] = plate.upper()
    return seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("image_dirs", nargs="+", help="Directories of images to label")
    parser.add_argument("-o", "--output", default="data/labels.csv",
                        help="Output CSV path (default: data/labels.csv)")
    parser.add_argument("--seed-from", metavar="EVAL_JSON",
                        help="Pre-fill true_plate from a previous eval_report.json")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing output file")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"error: '{args.output}' already exists.", file=sys.stderr)
        print("Labels are expensive to recreate; refusing to overwrite.", file=sys.stderr)
        print("Pass --force if you really mean it.", file=sys.stderr)
        return 2

    images = collect_images(args.image_dirs)
    if not images:
        print(f"error: no images found in {args.image_dirs}", file=sys.stderr)
        return 2

    seed = load_seed(args.seed_from) if args.seed_from else {}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seeded_count = 0
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "true_plate", "notes"])
        for name in images:
            plate = seed.get(name, "")
            note = SEED_MARKER if plate else ""
            if plate:
                seeded_count += 1
            writer.writerow([name, plate, note])

    print(f"Wrote {len(images)} rows to {args.output}")

    if seed:
        print(f"  {seeded_count} rows pre-filled and marked {SEED_MARKER}")
        print(f"  {len(images) - seeded_count} rows left blank")
        print()
        print(f"  Every seeded value is a MODEL PREDICTION, not ground truth.")
        print(f"  Open each image, confirm the plate, then delete the {SEED_MARKER} marker.")
        print(f"  Rows still marked when you run eval.py are counted and reported.")
    else:
        print("  All rows blank — fill in true_plate by hand.")

    print()
    print("Reminders (see LABELLING.md):")
    print("  - Empty true_plate means NO LEGIBLE PLATE. Those rows are the")
    print("    false-positive test set and they are the ones that matter most.")
    print("  - Aim for 15-20% of the corpus to be legitimately empty.")
    print("  - Transcribe what is painted, not what the format says it should be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
