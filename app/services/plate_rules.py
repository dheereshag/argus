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

# Re-exports for backwards compatibility
__all__ = [
    "CHAR_TO_DIGIT",
    "DIGIT_TO_CHAR",
    "INDIAN_PLATE_REGEX",
    "NON_PLATE_WORDS",
    "SERIES_CORRECTIONS",
    "STATE_CODES",
    "STATE_PREFIX_CORRECTIONS",
    "is_decal_word",
    "normalize_candidate_strings",
    "parse_plate_info",
]


def is_decal_word(word: str) -> bool:
    """Check if a candidate string is a common commercial vehicle decal word."""
    return word in NON_PLATE_WORDS or any(w in word for w in ("CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT"))


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
        if len(cand) == 10:
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:6]
            serial = cand[6:10]

            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_corr = "".join(CHAR_TO_DIGIT.get(c, c) for c in dist)
            series_corr = SERIES_CORRECTIONS.get(series, "".join(DIGIT_TO_CHAR.get(c, c) for c in series))
            serial_corr = "".join(CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series_corr + serial_corr
            if corrected not in results:
                results.append(corrected)

            if dist_corr.startswith("4"):
                alt_corr = st_corr + "0" + dist_corr[1:] + series_corr + serial_corr
                if alt_corr not in results:
                    results.append(alt_corr)

        elif len(cand) == 9:
            st = cand[:2]
            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)

            dist_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_a = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            dist_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_b = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[3:5])
            num_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        elif len(cand) == 8:
            st = cand[:2]
            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)

            dist_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_a = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[3:4])
            num_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[4:8])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            dist_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_b = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:8])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        if "BH" in cand:
            idx = cand.find("BH")
            if idx >= 2 and len(cand) >= idx + 6:
                yr = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[idx - 2 : idx])
                serial = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[idx + 2 : idx + 6])
                ser = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[idx + 6 :])
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

    if match.group(1):
        state_code = match.group(1).upper()
        state_name = STATE_CODES.get(state_code, "Unknown State")
        return {"plate": matched_plate, "state": state_name}

    if match.group(5):
        return {"plate": matched_plate, "state": STATE_CODES.get("BH", "Bharat Series (National)")}

    return {"plate": matched_plate, "state": "Unknown State"}
