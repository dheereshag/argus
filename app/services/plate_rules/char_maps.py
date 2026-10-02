"""Character substitution helpers and OCR confusion maps for plate normalization."""

CHAR_TO_DIGIT: dict[str, str] = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "H": "1",
    "Z": "2", "A": "4", "S": "5", "E": "6", "G": "6", "T": "7", "B": "8",
}

DIGIT_TO_CHAR: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "3": "J", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}

SERIES_CORRECTIONS: dict[str, str] = {
    "G3": "GJ",
    "D3": "DJ",
}


def apply_char_map(text: str, mapping: dict[str, str]) -> str:
    """Substitute characters in a string based on a substitution mapping dictionary."""
    return "".join(mapping.get(c, c) for c in text)
