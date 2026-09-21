"""Validate candidate plate strings against Indian regex and resolve State/UT."""

import re
from typing import Any

from app.constants import (
    HSRP_PREFIXES,
    INDIAN_PLATE_REGEX,
    STATE_CODES,
    STATE_PREFIX_CORRECTIONS,
)


def parse_plate_info(raw_plate: str | None) -> dict[str, Any] | None:
    """Validate candidate plate string against Indian plate regex and resolve State/UT."""
    if not raw_plate:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw_plate)).upper()
    if not cleaned:
        return None

    for pfx in HSRP_PREFIXES:
        if cleaned.startswith(pfx) and len(cleaned) >= len(pfx) + 5:
            cleaned = cleaned[len(pfx) :]
            break

    prefix_2 = cleaned[:2]
    if prefix_2 in STATE_PREFIX_CORRECTIONS:
        fixed = STATE_PREFIX_CORRECTIONS[prefix_2] + cleaned[2:]
        if INDIAN_PLATE_REGEX.fullmatch(fixed):
            cleaned = fixed

    match = INDIAN_PLATE_REGEX.fullmatch(cleaned)
    if not match:
        return None

    state_name = "Unknown State"
    if match.group(1):
        state_name = STATE_CODES.get(match.group(1).upper(), "Unknown State")
    elif match.group(8):
        state_name = STATE_CODES.get(match.group(8).upper(), "Unknown State")
    elif match.group(5) == "BH" or match.group(4):
        state_name = STATE_CODES.get("BH", "Bharat Series (National)")
    elif match.group(15):
        state_name = "Diplomatic Corps"
    elif match.group(10):
        state_name = "Military / Defence Series"

    return {"plate": cleaned, "state": state_name}
