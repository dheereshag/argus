"""Candidate plate collection, pairing evaluation, and ranking."""

from collections.abc import Callable
from typing import Any

from app.constants import INDIAN_PLATE_REGEX
from app.schemas import OCRToken, PlateCandidate
from app.services.ocr.spatial import compute_token_bounds
from app.services.plate_rules import normalize_candidate_strings
from app.services.plate_rules import parse_plate_info as default_parse


def collect_candidates(
    clean_tokens: list[OCRToken],
    lines: list[list[OCRToken]],
    pairs: list[tuple[float, str, float, list[OCRToken]]],
    raw_summary: str,
    parser_fn: Callable[[str | None], dict[str, Any] | None] | None = None,
) -> list[PlateCandidate]:
    """Evaluate single tokens, horizontal line merges, stacked lines, and spatial pairs."""
    candidates: list[PlateCandidate] = []
    seen: set[str] = set()
    parse_fn = parser_fn or default_parse

    def _eval(text: str, toks: list[OCRToken], y: float) -> None:
        conf = round(sum(t.score for t in toks) / max(len(toks), 1), 4) if toks else 0.0
        box = compute_token_bounds(toks)
        for rank, cand in enumerate(normalize_candidate_strings(text)):
            m = INDIAN_PLATE_REGEX.fullmatch(cand)
            if not m:
                continue
            info = parse_fn(m.group(0))
            if not info or info.get("plate") in seen:
                continue
            seen.add(info["plate"])
            info.update({"raw_text": raw_summary, "confidence": conf, "box": box})
            candidates.append(PlateCandidate(y_pos=y, rank=rank, info=info, confidence=conf, box=box))

    for tok in clean_tokens:
        _eval(tok.text, [tok], tok.cy or 0.0)
    for line in lines:
        if len(line) > 1:
            _eval("".join(t.text for t in line), line, line[0].cy or 0.0)
    for i in range(len(lines) - 1):
        _eval("".join(t.text for t in lines[i]) + "".join(t.text for t in lines[i + 1]), lines[i] + lines[i + 1], lines[i][0].cy or 0.0)
        if i + 2 < len(lines) and ((lines[i + 2][0].cy or 0.0) - (lines[i][0].cy or 0.0)) < 120.0:
            _eval("".join(t.text for t in lines[i]) + "".join(t.text for t in lines[i + 2]), lines[i] + lines[i + 2], lines[i][0].cy or 0.0)
    for _, p_text, y, p_toks in pairs:
        _eval(p_text, p_toks, y)
    return candidates
