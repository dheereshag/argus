"""Length-specific positional character substitution normalizers for Indian plates."""

from app.constants import CHAR_TO_DIGIT, DIGIT_TO_CHAR, SERIES_CORRECTIONS
from app.services.plate_rules.char_maps import apply_char_map


def sanitize_series(ser: str) -> str:
    """Replace MoRTH-prohibited letters 'I'->'J' and 'O'->'D' in plate series."""
    return ser.replace("I", "J").replace("O", "D")


def normalize_11_char(cand: str, st: str) -> list[str]:
    """Normalize 11-character plate with 3-letter series: SS DD AAA NNNN."""
    d = apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    s = apply_char_map(cand[4:7], DIGIT_TO_CHAR)
    res = [st + d + s + apply_char_map(cand[7:11], CHAR_TO_DIGIT)]
    if d.startswith("4"):
        res.append(st + "0" + d[1:] + s + res[0][len(st + d + s) :])
    if "I" in s or "O" in s:
        res.append(st + d + sanitize_series(s) + res[0][len(st + d + s) :])
    return res


def normalize_10_char(cand: str, st: str) -> list[str]:
    """Normalize standard 10-character plate: SS DD AA NNNN."""
    d = apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    s = SERIES_CORRECTIONS.get(cand[4:6], apply_char_map(cand[4:6], DIGIT_TO_CHAR))
    res = [st + d + s + apply_char_map(cand[6:10], CHAR_TO_DIGIT)]
    if d.startswith("4"):
        res.append(st + "0" + d[1:] + s + res[0][len(st + d + s) :])
    if "I" in s or "O" in s:
        res.append(st + d + sanitize_series(s) + res[0][len(st + d + s) :])
    return res


def normalize_9_char(cand: str, st: str) -> list[str]:
    """Normalize 9-character plate permutations."""
    cfgs = [
        (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
        (cand[2:3], CHAR_TO_DIGIT, cand[3:5], DIGIT_TO_CHAR, cand[5:9], CHAR_TO_DIGIT),
        (cand[2:4], CHAR_TO_DIGIT, cand[4:6], DIGIT_TO_CHAR, cand[6:9], CHAR_TO_DIGIT),
    ]
    res = [st + apply_char_map(d, dm) + apply_char_map(s, sm) + apply_char_map(n, nm) for d, dm, s, sm, n, nm in cfgs]
    for v in list(res):
        if len(v) == 9 and ("I" in v[4:6] or "O" in v[4:6]):
            res.append(v[:4] + sanitize_series(v[4:6]) + v[6:])
    return res


def normalize_8_char(cand: str, st: str) -> list[str]:
    """Normalize older 8-character plate permutations."""
    cfgs = [
        (cand[2:3], CHAR_TO_DIGIT, cand[3:4], DIGIT_TO_CHAR, cand[4:8], CHAR_TO_DIGIT),
        (cand[2:4], CHAR_TO_DIGIT, cand[4:5], DIGIT_TO_CHAR, cand[5:8], CHAR_TO_DIGIT),
        (cand[2:3], CHAR_TO_DIGIT, cand[3:5], DIGIT_TO_CHAR, cand[5:8], CHAR_TO_DIGIT),
    ]
    res = [st + apply_char_map(d, dm) + apply_char_map(s, sm) + apply_char_map(n, nm) for d, dm, s, sm, n, nm in cfgs]
    res.append(st + apply_char_map(cand[2:4], CHAR_TO_DIGIT) + apply_char_map(cand[4:8], CHAR_TO_DIGIT))
    return res
