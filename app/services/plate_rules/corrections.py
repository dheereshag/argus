"""OCR visual misread corrections for 2-character Indian state prefixes."""

STATE_PREFIX_CORRECTIONS: dict[str, str] = {
    "W8": "WB", "RT": "RJ", "R3": "RJ", "D1": "DL", "D7": "DL",
    "H8": "HR", "0D": "OD", "0R": "OR", "00": "OD", "0L": "DL",
    "K1": "KL", "T1": "TN", "A1": "AP", "VB": "WB", "NB": "WB",
    "2B": "WB", "MB": "WB", "38": "JH", "28": "JH", "M4": "MH",
    "C6": "CG", "K4": "KA", "P8": "PB", "G1": "GJ", "T5": "TS",
    "7S": "TS", "7G": "TG", "TC": "TG", "OL": "DL", "QL": "DL",
    "1L": "DL", "RA": "KA", "NP": "MP", "M0": "MP", "PJ": "RJ",
    "0P": "UP", "OP": "UP", "7N": "TN", "1N": "TN",
}
