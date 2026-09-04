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
    """Check if a candidate string is a common commercial vehicle decal word."""
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


def _apply_char_map(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in text)


def normalize_candidate_strings(raw_str: str) -> list[str]:
    """Generate normalized plate candidate variants using positional character rules for Indian plates."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 6:
        return []

    candidates = [cleaned]
    for prefix, repl in STATE_PREFIX_CORRECTIONS.items():
        if cleaned.startswith(prefix):
            candidates.append(repl + cleaned[len(prefix) :])

    results = list(candidates)

    for cand in candidates:
        length = len(cand)
        st_corr = STATE_PREFIX_CORRECTIONS.get(cand[:2], cand[:2])

        # Standard 10-char plates: SS DD AA NNNN
        if length == 10:
            dist = _apply_char_map(cand[2:4], CHAR_TO_DIGIT)
            ser = SERIES_CORRECTIONS.get(cand[4:6], _apply_char_map(cand[4:6], DIGIT_TO_CHAR))
            num = _apply_char_map(cand[6:10], CHAR_TO_DIGIT)
            for c in (st_corr + dist + ser + num, st_corr + "0" + dist[1:] + ser + num if dist.startswith("4") else None):
                if c and c not in results:
                    results.append(c)

        # 9-char plates: SS D AA NNNN or SS DD A NNNN
        elif length == 9:
            configs = [
                (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
                (cand[2:3], CHAR_TO_DIGIT, cand[3:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
            ]
            for d_raw, d_map, s_raw, s_map, n_raw, n_map in configs:
                c = st_corr + _apply_char_map(d_raw, d_map) + _apply_char_map(s_raw, s_map) + _apply_char_map(n_raw, n_map)
                if c not in results:
                    results.append(c)

        # 8-char plates: SS D A NNNN or SS DD A NNN
        elif length == 8:
            configs = [
                (cand[2:3], CHAR_TO_DIGIT, cand[3:4], DIGIT_TO_CHAR, cand[4:8], CHAR_TO_DIGIT),
                (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:8], CHAR_TO_DIGIT),
            ]
            for d_raw, d_map, s_raw, s_map, n_raw, n_map in configs:
                c = st_corr + _apply_char_map(d_raw, d_map) + _apply_char_map(s_raw, s_map) + _apply_char_map(n_raw, n_map)
                if c not in results:
                    results.append(c)

        # Bharat series: YY BH NNNN AA
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
    """Validate candidate plate string against Indian plate regex and resolve State/UT."""
    if not raw_plate:
        return None

    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate)).upper()
    if not cleaned:
        return None

    if cleaned.startswith("W8"):
        cleaned = "WB" + cleaned[2:]

    match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
    if not match:
        return None

    matched_plate = cleaned
    state_name = "Unknown State"

    if match.group(1):
        state_code = match.group(1).upper()
        state_name = STATE_CODES.get(state_code, "Unknown State")
    elif match.group(5):
        state_name = STATE_CODES.get("BH", "Bharat Series (National)")

    return {"plate": matched_plate, "state": state_name}
