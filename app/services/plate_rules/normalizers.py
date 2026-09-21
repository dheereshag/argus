"""Length-specific positional character substitution normalizers for Indian plates."""

from app.constants import CHAR_TO_DIGIT, DIGIT_TO_CHAR, SERIES_CORRECTIONS
from app.services.plate_rules.char_maps import apply_char_map


def normalize_11_char(cand: str, st: str) -> list[str]:
    """Normalize 11-character plate with 3-letter series: SS DD AAA NNNN."""
    dist = apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    ser = apply_char_map(cand[4:7], DIGIT_TO_CHAR)
    num = apply_char_map(cand[7:11], CHAR_TO_DIGIT)
    res = [st + dist + ser + num]
    if dist.startswith("4"):
        res.append(st + "0" + dist[1:] + ser + num)
    if "I" in ser or "O" in ser:
        res.append(st + dist + ser.replace("I", "J").replace("O", "D") + num)
    return res


def normalize_10_char(cand: str, st: str) -> list[str]:
    """Normalize standard 10-character plate: SS DD AA NNNN."""
    dist = apply_char_map(cand[2:4], CHAR_TO_DIGIT)
    ser = SERIES_CORRECTIONS.get(cand[4:6], apply_char_map(cand[4:6], DIGIT_TO_CHAR))
    num = apply_char_map(cand[6:10], CHAR_TO_DIGIT)
    res = [st + dist + ser + num]
    if dist.startswith("4"):
        res.append(st + "0" + dist[1:] + ser + num)
    if "I" in ser or "O" in ser:
        res.append(st + dist + ser.replace("I", "J").replace("O", "D") + num)
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
        if len(v) == 9 and v[4] in ("I", "O"):
            res.append(v[:4] + ("J" if v[4] == "I" else "D") + v[5:])
        ser2 = v[4:6]
        if len(v) == 9 and ("I" in ser2 or "O" in ser2):
            res.append(v[:4] + ser2.replace("I", "J").replace("O", "D") + v[6:])
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
