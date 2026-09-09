"""
Indian vehicle registration number normalization and validation heuristics.

Applies domain knowledge of Indian license plate formats to correct common optical
character recognition (OCR) errors based on positional syntax (digits vs letters):
  - Standard Format: SS DD AA NNNN / SS DD AAA NNNN (State, District RTO, Series, Number)
  - Bharat (BH) Format: YY BH NNNN AA (Year, National BH Code, Number, Series)
"""

import re
from typing import Any

from app.constants import (
    CHAR_TO_DIGIT,
    DIGIT_TO_CHAR,
    INDIAN_PLATE_REGEX,
    NON_PLATE_WORDS,
    SERIES_CORRECTIONS,
    STATE_CODES,
    STATE_PREFIX_CORRECTIONS,
)

__all__ = [
    "is_decal_word",
    "is_phone_number",
    "normalize_candidate_strings",
    "parse_plate_info",
]

_PHONE_PATTERN: re.Pattern[str] = re.compile(r"^[6-9]\d{9}$")


def is_phone_number(text: str) -> bool:
    """Check if a string matches a 10-digit Indian mobile telephone number."""
    return bool(_PHONE_PATTERN.match(text))


def is_decal_word(word: str) -> bool:
    """
    Check if a candidate text matches commercial vehicle decals, badges, or phone numbers.

    Trucks and buses in India frequently feature painted decals ('GOODS CARRIER',
    'NATIONAL PERMIT'), manufacturer badges ('TATA', 'LEYLAND'), or driver contact numbers.
    Filtering these early prevents them from being misinterpreted as registration plates.

    Args:
        word: Normalized uppercase alphanumeric token string.

    Returns:
        bool: True if the word matches known commercial decals, blacklisted words, or mobile numbers.
    """
    if word in NON_PLATE_WORDS or is_phone_number(word):
        return True
    return any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT", "FASTAG", "DIESEL"))


def _apply_char_map(text: str, mapping: dict[str, str]) -> str:
    """Substitute characters in a string based on a substitution mapping dictionary."""
    return "".join(mapping.get(c, c) for c in text)


def _normalize_11_char(cand: str, st_corr: str) -> list[str]:
    """Normalize 11-character plate with 3-letter series: SS DD AAA NNNN."""
    dist = _apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    ser = _apply_char_map(cand[4:7], DIGIT_TO_CHAR)
    num = _apply_char_map(cand[7:11], CHAR_TO_DIGIT)
    variants = [st_corr + dist + ser + num]
    if dist.startswith("4"):
        variants.append(st_corr + "0" + dist[1:] + ser + num)
    return variants


def _normalize_10_char(cand: str, st_corr: str) -> list[str]:
    """Normalize standard 10-character plate: SS DD AA NNNN."""
    dist = _apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    ser = SERIES_CORRECTIONS.get(cand[4:6], _apply_char_map(cand[4:6], DIGIT_TO_CHAR))
    num = _apply_char_map(cand[6:10], CHAR_TO_DIGIT)
    variants = [st_corr + dist + ser + num]
    if dist.startswith("4"):
        variants.append(st_corr + "0" + dist[1:] + ser + num)
    return variants


def _normalize_9_char(cand: str, st_corr: str) -> list[str]:
    """Normalize 9-character plate permutations (SS D AA NNNN or SS DD A NNNN)."""
    configs = [
        (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
        (cand[2:3], CHAR_TO_DIGIT, cand[3:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
    ]
    return [
        st_corr + _apply_char_map(d, d_map) + _apply_char_map(s, s_map) + _apply_char_map(n, n_map)
        for d, d_map, s, s_map, n, n_map in configs
    ]


def _normalize_8_char(cand: str, st_corr: str) -> list[str]:
    """Normalize older 8-character plate permutations (SS D A NNNN or SS DD A NNN)."""
    configs = [
        (cand[2:3], CHAR_TO_DIGIT, cand[3:4], DIGIT_TO_CHAR, cand[4:8], CHAR_TO_DIGIT),
        (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:8], CHAR_TO_DIGIT),
    ]
    return [
        st_corr + _apply_char_map(d, d_map) + _apply_char_map(s, s_map) + _apply_char_map(n, n_map)
        for d, d_map, s, s_map, n, n_map in configs
    ]


def _normalize_bh_series(cand: str) -> str | None:
    """Normalize Bharat (BH) Series format: YY BH NNNN AA."""
    fixed = cand
    for sub in ("8H", "6H"):
        if sub in fixed and "BH" not in fixed:
            fixed = fixed.replace(sub, "BH", 1)
    if "BH" not in fixed:
        return None
    idx = fixed.find("BH")
    if idx >= 2 and len(fixed) >= idx + 6:
        yr = _apply_char_map(fixed[idx - 2 : idx], CHAR_TO_DIGIT)
        serial = _apply_char_map(fixed[idx + 2 : idx + 6], CHAR_TO_DIGIT)
        ser = _apply_char_map(fixed[idx + 6 :], DIGIT_TO_CHAR)
        return yr + "BH" + serial + ser
    return None


def _expand_candidates_for_string(cand: str, results: list[str]) -> None:
    """Generate and append positional character permutations for a single candidate string."""
    st_corr = STATE_PREFIX_CORRECTIONS.get(cand[:2], cand[:2])
    generated: list[str] = []

    if len(cand) == 11:
        generated.extend(_normalize_11_char(cand, st_corr))
    elif len(cand) == 10:
        generated.extend(_normalize_10_char(cand, st_corr))
    elif len(cand) == 9:
        generated.extend(_normalize_9_char(cand, st_corr))
    elif len(cand) == 8:
        generated.extend(_normalize_8_char(cand, st_corr))

    bh_cand = _normalize_bh_series(cand)
    if bh_cand:
        generated.append(bh_cand)

    for item in generated:
        if item not in results:
            results.append(item)


def normalize_candidate_strings(raw_str: str) -> list[str]:
    """
    Generate normalized plate candidate variants using positional character rules for Indian plates.

    Handles standard 8-11 character formats, HSRP 'IND' prefix stripping, and Bharat (BH) series.

    Args:
        raw_str: Unnormalized OCR text string.

    Returns:
        list[str]: Ranked list of synthesized candidate strings to test against the regex.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 6:
        return []

    candidates = [cleaned]
    # Strip HSRP 'IND' national strip prefix if fused to registration mark
    if cleaned.startswith("IND") and len(cleaned) >= 8:
        candidates.append(cleaned[3:])

    for prefix, repl in STATE_PREFIX_CORRECTIONS.items():
        for base in list(candidates):
            if base.startswith(prefix):
                candidates.append(repl + base[len(prefix) :])

    results = list(candidates)
    for cand in candidates:
        _expand_candidates_for_string(cand, results)

    return results


def parse_plate_info(raw_plate: str | None) -> dict[str, Any] | None:
    """
    Validate candidate plate string against Indian plate regex and resolve State/UT.

    Args:
        raw_plate: Candidate string to validate.

    Returns:
        dict[str, Any] | None: Dictionary with keys 'plate' and 'state' if valid, None otherwise.
    """
    if not raw_plate:
        return None

    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate)).upper()
    if not cleaned:
        return None

    # Strip HSRP 'IND' prefix if present on candidate
    if cleaned.startswith("IND") and len(cleaned) >= 8:
        cleaned = cleaned[3:]

    # Handle common West Bengal OCR misread prefix
    if cleaned.startswith("W8"):
        cleaned = "WB" + cleaned[2:]

    match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
    if not match:
        return None

    matched_plate = cleaned
    state_name = "Unknown State"

    # Group 1 captures standard state prefix; Group 4 captures Bharat Series 'BH'
    if match.group(1):
        state_code = match.group(1).upper()
        state_name = STATE_CODES.get(state_code, "Unknown State")
    elif match.group(4):
        state_name = STATE_CODES.get("BH", "Bharat Series (National)")

    return {"plate": matched_plate, "state": state_name}

