"""Character substitution helpers for plate normalization."""


def apply_char_map(text: str, mapping: dict[str, str]) -> str:
    """Substitute characters in a string based on a substitution mapping dictionary."""
    return "".join(mapping.get(c, c) for c in text)
