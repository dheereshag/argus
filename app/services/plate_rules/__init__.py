"""Indian vehicle registration number normalization, validation, and domain heuristics."""

from app.services.plate_rules.expander import normalize_candidate_strings
from app.services.plate_rules.filters import is_decal_word, is_phone_number
from app.services.plate_rules.normalizers import (
    normalize_8_char as _normalize_8_char,
)
from app.services.plate_rules.normalizers import (
    normalize_9_char as _normalize_9_char,
)
from app.services.plate_rules.parser import parse_plate_info
from app.services.plate_rules.resolver import resolve_plate_from_text

__all__ = [
    "_normalize_8_char",
    "_normalize_9_char",
    "is_decal_word",
    "is_phone_number",
    "normalize_candidate_strings",
    "parse_plate_info",
    "resolve_plate_from_text",
]
