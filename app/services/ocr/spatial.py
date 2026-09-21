"""2D token line clustering and bounding box computation."""

from app.schemas import OCRToken


def is_same_horizontal_line(t1: OCRToken, t2: OCRToken) -> bool:
    """Check if two tokens share the same horizontal text line via box overlap or centroid."""
    if t1.box is not None and t2.box is not None:
        overlap = max(0, min(t1.box[3], t2.box[3]) - max(t1.box[1], t2.box[1]))
        return (overlap / min(max(1, t1.box[3] - t1.box[1]), max(1, t2.box[3] - t2.box[1]))) >= 0.4
    if t1.cy is not None and t2.cy is not None:
        return abs(t1.cy - t2.cy) <= 8.0
    return False


def cluster_horizontal_lines(tokens: list[OCRToken]) -> list[list[OCRToken]]:
    """Group tokens sharing similar vertical Y coordinates into horizontal lines."""
    if not tokens:
        return []
    sorted_y = sorted(tokens, key=lambda t: t.cy if t.cy is not None else 9999.0)
    lines: list[list[OCRToken]] = []
    for tok in sorted_y:
        placed = False
        for line in lines:
            if is_same_horizontal_line(line[0], tok):
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])
    for line in lines:
        line.sort(key=lambda t: t.cx if t.cx is not None else 0.0)
    return lines


def compute_token_bounds(tokens: list[OCRToken]) -> tuple[int, int, int, int] | None:
    """Compute the enclosing (x1, y1, x2, y2) bounding box across a set of tokens."""
    valid = [t.box for t in tokens if t.box is not None]
    return (min(b[0] for b in valid), min(b[1] for b in valid), max(b[2] for b in valid), max(b[3] for b in valid)) if valid else None
