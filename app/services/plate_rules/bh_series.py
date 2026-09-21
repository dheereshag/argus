"""Bharat (BH) Series format normalization."""

from app.constants import CHAR_TO_DIGIT, DIGIT_TO_CHAR
from app.services.plate_rules.char_maps import apply_char_map


def normalize_bh_series(cand: str) -> str | None:
    """Normalize Bharat (BH) Series format: YY BH NNNN AA."""
    fixed = cand
    for sub in ("8H", "6H"):
        if sub in fixed and "BH" not in fixed:
            fixed = fixed.replace(sub, "BH", 1)
    if "BH" not in fixed:
        return None
    idx = fixed.find("BH")
    if idx >= 2 and len(fixed) >= idx + 6:
        yr = apply_char_map(fixed[idx - 2 : idx], CHAR_TO_DIGIT)
        serial = apply_char_map(fixed[idx + 2 : idx + 6], CHAR_TO_DIGIT)
        ser = apply_char_map(fixed[idx + 6 :], DIGIT_TO_CHAR)
        return yr + "BH" + serial + ser
    return None
