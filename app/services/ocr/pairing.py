"""Construct 2-line spatial candidate pairings sorted by Euclidean centroid distance."""

import math

from app.schemas import OCRToken


def build_spatial_pairs(clean_tokens: list[OCRToken]) -> list[tuple[float, str, float, list[OCRToken]]]:
    """Construct 2-line spatial candidate pairings sorted by Euclidean centroid distance."""
    candidate_pairs: list[tuple[float, str, float, list[OCRToken]]] = []
    n = len(clean_tokens)
    for i in range(n):
        tok_a = clean_tokens[i]
        for j in range(i + 1, min(i + 6, n)):
            tok_b = clean_tokens[j]
            if tok_a.cx is not None and tok_a.cy is not None and tok_b.cx is not None and tok_b.cy is not None:
                dist = math.hypot(tok_a.cx - tok_b.cx, tok_a.cy - tok_b.cy)
                y_mean = float((tok_a.cy + tok_b.cy) / 2.0)
                top_tok, bot_tok = (tok_a, tok_b) if tok_a.cy <= tok_b.cy else (tok_b, tok_a)
            else:
                dist = float(abs(i - j) * 100.0)
                y_mean = tok_a.cy or tok_b.cy or 0.0
                top_tok, bot_tok = tok_a, tok_b

            candidate_pairs.append((dist, top_tok.text + bot_tok.text, y_mean, [top_tok, bot_tok]))
            if abs((tok_a.cy or 0.0) - (tok_b.cy or 0.0)) < 10.0:
                candidate_pairs.append((dist + 0.1, bot_tok.text + top_tok.text, y_mean, [bot_tok, top_tok]))

    candidate_pairs.sort(key=lambda p: p[0])
    return candidate_pairs
