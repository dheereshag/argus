"""Generate candidate string permutations using positional rules for Indian plates."""

import re

from app.constants import HSRP_PREFIXES, STATE_PREFIX_CORRECTIONS
from app.services.plate_rules.bh_series import normalize_bh_series
from app.services.plate_rules.normalizers import (
    normalize_8_char,
    normalize_9_char,
    normalize_10_char,
    normalize_11_char,
)

_NORM_BY_LEN = ((11, normalize_11_char), (10, normalize_10_char), (9, normalize_9_char), (8, normalize_8_char))


def _expand(cand: str, results: list[str]) -> None:
    st = STATE_PREFIX_CORRECTIONS.get(cand[:2], cand[:2])
    gen: list[str] = []
    for t_len, norm in _NORM_BY_LEN:
        if len(cand) == t_len:
            gen.extend(norm(cand, st))
            break
    bh = normalize_bh_series(cand)
    if bh:
        gen.append(bh)
    for item in gen:
        if item not in results:
            results.append(item)


def normalize_candidate_strings(raw_str: str) -> list[str]:
    """Generate normalized plate candidate variants using positional character rules."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 5:
        return []

    cands = [cleaned]
    for pfx in HSRP_PREFIXES:
        if cleaned.startswith(pfx) and len(cleaned) >= len(pfx) + 5:
            cands.append(cleaned[len(pfx) :])

    if len(cleaned) in (10, 11) and cleaned[0].isalpha() and cleaned[1:3].isdigit() and cleaned[3].isalpha():
        cands.append(cleaned[1:])

    for pfx, repl in STATE_PREFIX_CORRECTIONS.items():
        for base in list(cands):
            if base.startswith(pfx):
                cands.append(repl + base[len(pfx) :])

    res = list(cands)
    for cand in cands:
        _expand(cand, res)
    return res
