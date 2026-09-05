"""
Indian vehicle registration number normalization and validation heuristics.

Applies domain knowledge of Indian license plate formats to correct common optical
character recognition (OCR) errors based on positional syntax (digits vs letters):
  - Standard Format: SS DD AA NNNN (State, District RTO, Series, Number)
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
    "normalize_candidate_strings",
    "parse_plate_info",
]


def is_decal_word(word: str) -> bool:
    """
    Check if a candidate text string matches common commercial vehicle decal words.

    Trucks and buses in India frequently feature prominent painted decals such as
    'GOODS CARRIER', 'NATIONAL PERMIT', or manufacturer badges ('TATA', 'LEYLAND').
    Filtering these early prevents them from being misinterpreted as registration plates.

    Args:
        word: Normalized uppercase alphanumeric token string.

    Returns:
        bool: True if the word matches known commercial decals or blacklisted keywords.
    """
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


def _apply_char_map(text: str, mapping: dict[str, str]) -> str:
    """
    Substitute characters in a string based on a substitution mapping dictionary.

    Args:
        text: Input string to transform.
        mapping: Dictionary mapping source characters to target characters.

    Returns:
        str: Transformed string with mapped character replacements applied.
    """
    return "".join(mapping.get(c, c) for c in text)


def normalize_candidate_strings(raw_str: str) -> list[str]:
    """
    Generate normalized plate candidate variants using positional character rules for Indian plates.

    Indian license plates follow strict positional character rules:
      - Positions 0..1: State prefix (always 2 alphabetic letters, e.g., 'DL', 'MH')
      - Positions 2..3: District RTO code (numeric digits, e.g., '01', '12')
      - Following 1-3 chars: Series code (alphabetic letters, e.g., 'A', 'AB', 'GA')
      - Trailing 3-4 chars: Sequential registration number (numeric digits, e.g., '0165', '1234')

    This function tests multiple permutations by correcting characters based on their expected
    positional type (e.g. converting 'O'->'0' in digit slots, '0'->'O' in letter slots).

    Args:
        raw_str: Unnormalized OCR text string.

    Returns:
        list[str]: Ranked list of synthesized candidate strings to test against the regex.
    """
    # Remove all punctuation, whitespace, and non-alphanumeric characters
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 6:
        return []

    candidates = [cleaned]

    # Check for known optical OCR mistakes in state code prefixes (e.g., 'W8' -> 'WB', 'D1' -> 'DL')
    for prefix, repl in STATE_PREFIX_CORRECTIONS.items():
        if cleaned.startswith(prefix):
            candidates.append(repl + cleaned[len(prefix) :])

    results = list(candidates)

    for cand in candidates:
        length = len(cand)
        st_corr = STATE_PREFIX_CORRECTIONS.get(cand[:2], cand[:2])

        # ----------------------------------------------------------------------
        # Case 1: Standard 10-character plate: SS DD AA NNNN
        # Example: RJ 09 GA 0165 -> (State: RJ, Dist: 09, Ser: GA, Num: 0165)
        # ----------------------------------------------------------------------
        if length == 10:
            dist = _apply_char_map(cand[2:4], CHAR_TO_DIGIT)
            ser = SERIES_CORRECTIONS.get(cand[4:6], _apply_char_map(cand[4:6], DIGIT_TO_CHAR))
            num = _apply_char_map(cand[6:10], CHAR_TO_DIGIT)

            # Heuristic for single-digit district RTO where '0' was misread as '4' (e.g. '49' -> '09')
            for c in (st_corr + dist + ser + num, st_corr + "0" + dist[1:] + ser + num if dist.startswith("4") else None):
                if c and c not in results:
                    results.append(c)

        # ----------------------------------------------------------------------
        # Case 2: 9-character plate: SS D AA NNNN or SS DD A NNNN
        # Example: DL 1 CA 1234 or MH 12 A 5678
        # ----------------------------------------------------------------------
        elif length == 9:
            configs = [
                # Configuration A: 2-digit district + 1-letter series
                (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
                # Configuration B: 1-digit district + 2-letter series
                (cand[2:3], CHAR_TO_DIGIT, cand[3:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
            ]
            for d_raw, d_map, s_raw, s_map, n_raw, n_map in configs:
                c = st_corr + _apply_char_map(d_raw, d_map) + _apply_char_map(s_raw, s_map) + _apply_char_map(n_raw, n_map)
                if c not in results:
                    results.append(c)

        # ----------------------------------------------------------------------
        # Case 3: 8-character plate: SS D A NNNN or SS DD A NNN
        # Older or regional format plates (e.g. DL 1 A 1234)
        # ----------------------------------------------------------------------
        elif length == 8:
            configs = [
                (cand[2:3], CHAR_TO_DIGIT, cand[3:4], DIGIT_TO_CHAR, cand[4:8], CHAR_TO_DIGIT),
                (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:8], CHAR_TO_DIGIT),
            ]
            for d_raw, d_map, s_raw, s_map, n_raw, n_map in configs:
                c = st_corr + _apply_char_map(d_raw, d_map) + _apply_char_map(s_raw, s_map) + _apply_char_map(n_raw, n_map)
                if c not in results:
                    results.append(c)

        # ----------------------------------------------------------------------
        # Case 4: Bharat (BH) Series format: YY BH NNNN AA
        # Example: 22 BH 1234 AA
        # ----------------------------------------------------------------------
        if "BH" in cand:
            idx = cand.find("BH")
            if idx >= 2 and len(cand) >= idx + 6:
                yr = _apply_char_map(cand[idx - 2 : idx], CHAR_TO_DIGIT)
                serial = _apply_char_map(cand[idx + 2 : idx + 6], CHAR_TO_DIGIT)
                ser = _apply_char_map(cand[idx + 6 :], DIGIT_TO_CHAR)
                bh_cand = yr + "BH" + serial + ser
                if bh_cand not in results:
                    results.append(bh_cand)

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

    # Handle common West Bengal OCR misread prefix
    if cleaned.startswith("W8"):
        cleaned = "WB" + cleaned[2:]

    match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
    if not match:
        return None

    matched_plate = cleaned
    state_name = "Unknown State"

    # Group 1 captures standard state prefix; Group 5 captures Bharat Series 'BH'
    if match.group(1):
        state_code = match.group(1).upper()
        state_name = STATE_CODES.get(state_code, "Unknown State")
    elif match.group(5):
        state_name = STATE_CODES.get("BH", "Bharat Series (National)")

    return {"plate": matched_plate, "state": state_name}
