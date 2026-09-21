"""Commercial vehicle decal and driver contact phone number filtering."""

import re

from app.constants import COMMERCIAL_DECAL_SUBSTRINGS, NON_PLATE_WORDS

_CONTACT_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:MOB|MOBILE|PH|PHONE|TEL|CALL|CONTACT)?([6-9]\d{9})$"
)


def is_phone_number(text: str) -> bool:
    """Check if a string matches a 10-digit Indian mobile telephone number."""
    return bool(_CONTACT_PREFIX_PATTERN.match(text))


def is_decal_word(word: str) -> bool:
    """Check if candidate text matches painted decals, badges, or phone numbers."""
    if word in NON_PLATE_WORDS or is_phone_number(word):
        return True
    return any(w in word for w in COMMERCIAL_DECAL_SUBSTRINGS)
