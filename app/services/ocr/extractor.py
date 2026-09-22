"""Extract license plates from image using OCR tokens, spatial lines, and candidates."""

from typing import Any

from PIL import Image

from app.services.ocr.candidates import collect_candidates
from app.services.ocr.engine import run_ocr_inference
from app.services.ocr.pairing import build_spatial_pairs
from app.services.ocr.spatial import cluster_horizontal_lines
from app.services.ocr.suppression import suppress_overlapping_candidates
from app.services.ocr.tokens import clean_and_filter_tokens


def extract_plates(img_pil: Image.Image, engine_getter: Any, parse_fn: Any) -> list[dict[str, Any]]:
    """Extract and validate Indian license plates from a PIL image array."""
    raw_items = run_ocr_inference(img_pil, engine_getter)
    if not raw_items:
        return []
    clean_tokens, raw_summary = clean_and_filter_tokens(raw_items)
    lines = cluster_horizontal_lines(clean_tokens)
    pairs = build_spatial_pairs(clean_tokens)
    candidates = collect_candidates(clean_tokens, lines, pairs, raw_summary, parse_fn)
    if candidates:
        candidates.sort(key=lambda c: (len(c.info.get("plate", "")) >= 10, -c.rank, len(c.info.get("plate", "")), c.confidence, c.y_pos), reverse=True)
        suppressed = suppress_overlapping_candidates(candidates)
        if suppressed:
            return [c.info for c in suppressed]
    return [{"plate": "N/A", "state": "N/A", "raw_text": raw_summary, "confidence": None, "box": None}]

