"""Token filtering, cleaning, and text normalization."""

import re

from app.schemas import OCRToken
from app.services.plate_rules import is_decal_word


def clean_and_filter_tokens(raw_items: list[OCRToken]) -> tuple[list[OCRToken], str]:
    """Filter low-confidence tokens and decals, returning clean tokens and summary."""
    clean_tokens: list[OCRToken] = []
    raw_text_parts: list[str] = []
    seen: set[str] = set()
    for tok in raw_items:
        if not tok.text or tok.score < 0.20:
            continue
        raw_text_parts.append(tok.text.strip())
        for chunk in [tok.text, *tok.text.split()]:
            c = re.sub(r"[^A-Za-z0-9]", "", chunk).upper()
            if len(c) >= 2 and not is_decal_word(c) and c not in seen:
                seen.add(c)
                clean_tokens.append(OCRToken(text=c, score=tok.score, cx=tok.cx, cy=tok.cy, box=tok.box))
    return clean_tokens, " ".join(raw_text_parts) if raw_text_parts else "N/A"
