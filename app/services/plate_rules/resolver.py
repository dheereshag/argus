"""Unified raw OCR text normalization, decal filtering, and plate validation."""

from app.services.plate_rules.expander import normalize_candidate_strings
from app.services.plate_rules.filters import is_decal_word, is_phone_number
from app.services.plate_rules.parser import parse_plate_info


def resolve_plate_from_text(raw_text: str | None) -> str | None:
    """Filter decals, normalize common OCR confusions, and validate Indian plate."""
    if not raw_text or is_decal_word(raw_text) or is_phone_number(raw_text):
        return None
    for cand in normalize_candidate_strings(raw_text):
        if (info := parse_plate_info(cand)) and (p := info.get("plate")):
            return p
    return None
