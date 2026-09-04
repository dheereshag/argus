import re

# ==============================================================================
# Image Processing Constants & MIME Types
# ==============================================================================
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/pjpeg",
        "image/png",
        "image/x-png",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
    }
)

# ==============================================================================
# YOLO Object Detection Constants
# ==============================================================================
PERSON_CLASS_ID = 0
FOUR_WHEELER_CLASS_NAMES: dict[int, str] = {2: "car", 5: "bus", 7: "truck"}
MAX_DETECTIONS = 100
MIN_CROP_EDGE_PX = 8

# ==============================================================================
# Indian License Plate Domain Lookup Tables
# ==============================================================================
STATE_CODES: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "BP": "Police / Government Series",
    "BH": "Bharat Series (National)",
}

STATE_PREFIX_PATTERN = "|".join(sorted([k for k in STATE_CODES if k != "BH"], key=len, reverse=True))

INDIAN_PLATE_REGEX: re.Pattern[str] = re.compile(
    r"(?:"
    rf"({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d|[1-9])[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{3,4}})"
    r"|"
    r"(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})"
    r")",
    re.IGNORECASE,
)

CHAR_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "H": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

DIGIT_TO_CHAR: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

SERIES_CORRECTIONS: dict[str, str] = {
    "G3": "GJ",
    "GT": "GJ",
    "GI": "GJ",
    "GB": "GB",
    "D3": "DJ",
    "DT": "DJ",
    "DI": "DJ",
}

STATE_PREFIX_CORRECTIONS: dict[str, str] = {
    "W8": "WB",
    "RT": "RJ",
    "R3": "RJ",
    "D1": "DL",
    "D7": "DL",
    "H8": "HR",
    "0D": "OD",
    "0R": "OR",
    "00": "OD",
    "0L": "DL",
    "K1": "KL",
    "T1": "TN",
    "A1": "AP",
    "VB": "WB",
    "NB": "WB",
    "2B": "WB",
    "MB": "WB",
    "38": "JH",
    "28": "JH",
}

NON_PLATE_WORDS: frozenset[str] = frozenset(
    {
        "GOOD",
        "GOODS",
        "LUCK",
        "CARRIER",
        "SPEED",
        "TATA",
        "ASHOK",
        "LEYLAND",
        "EICHER",
        "INDIAN",
        "NATIONAL",
        "PERMIT",
        "DIESEL",
        "STOP",
        "HORN",
        "PLEASE",
        "FAST",
        "SUPER",
        "INDIA",
        "ROAD",
        "LINES",
        "TRANSPORT",
        "MOTORS",
        "SUPREME",
        "CEMENT",
        "COACH",
        "AIR",
        "BRAKE",
        "ALL",
        "STATE",
        "40KM",
        "PUBLIC",
        "AUTO",
        "SAFETY",
        "FIRST",
    }
)
