"""
Domain constants, lookup tables, and regular expressions for Argus ANPR.

This module provides:
  - Allowed image file formats and MIME types for payload validation.
  - YOLO object detection constants (COCO class IDs for vehicles and humans).
  - Indian license plate domain lookup tables: State/UT prefix codes, regex matching,
    character disambiguation maps (OCR confusion matrices), and decal blacklists.
"""

import re

# ==============================================================================
# Image Processing Constants & MIME Types
# ==============================================================================

# Permitted raster image formats parsed by Pillow / OpenCV
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "BMP"})

# Permitted MIME types in HTTP Content-Type headers for incoming upload validation
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

# Standard COCO dataset class index for 'person'
PERSON_CLASS_ID = 0

# Mapping of COCO class indices to 4-wheeler vehicle category names
# 2: car, 5: bus, 7: truck
FOUR_WHEELER_CLASS_NAMES: dict[int, str] = {2: "car", 5: "bus", 7: "truck"}

# Upper bound cap on raw detections evaluated per frame to prevent DoS from noisy inputs
MAX_DETECTIONS = 100

# Minimum bounding box edge dimension (in pixels) required to attempt a valid vehicle crop
MIN_CROP_EDGE_PX = 8

# ==============================================================================
# Indian License Plate Domain Lookup Tables
# ==============================================================================

# Mapping of 2-letter state/UT codes to their canonical full names.
# Includes special series:
#   - 'BP': Police / Government departmental vehicles
#   - 'BH': Bharat Series (inter-state non-transferable registration for defense & central employees)
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

# Regex prefix group created by joining state codes ordered by descending length
# to avoid premature partial prefix matches (excludes BH which follows a distinct pattern).
STATE_PREFIX_PATTERN = "|".join(sorted([k for k in STATE_CODES if k != "BH"], key=len, reverse=True))

# Compiled regular expression for Indian vehicle registration plates.
# Matches two primary structures:
# 1. Standard State Format:
#    (State Code) + (1-2 digit District RTO) + (1-3 letter Series) + (3-4 digit unique number)
#    Examples: MH12AB1234, DL01A5678, KA03MB100, RJ09GA0165
# 2. Bharat (BH) Series Format:
#    (2-digit Year) + BH + (4-digit number) + (1-2 letter Series)
#    Example: 22BH1234AA
INDIAN_PLATE_REGEX: re.Pattern[str] = re.compile(
    r"(?:"
    rf"({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d|[1-9])[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{3,4}})"
    r"|"
    r"(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})"
    r")",
    re.IGNORECASE,
)

# OCR visual confusion mappings: letters frequently misidentified in positions expected to be digits.
# e.g., 'O' or 'D' in number sequence -> '0', 'I' or 'L' -> '1', 'B' -> '8'
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

# OCR visual confusion mappings: digits frequently misidentified in positions expected to be alphabetic.
# e.g., '0' in series/state prefix -> 'O', '1' -> 'I', '8' -> 'B', '5' -> 'S'
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

# Common OCR errors in plate series substrings where letters are misread as digits/similar letters
SERIES_CORRECTIONS: dict[str, str] = {
    "G3": "GJ",
    "GT": "GJ",
    "GI": "GJ",
    "GB": "GB",
    "D3": "DJ",
    "DT": "DJ",
    "DI": "DJ",
}

# Common OCR errors in 2-character state prefixes (e.g. 'W8' for 'WB', 'D1' for 'DL', '0D' for 'OD')
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

# High-frequency text decals, manufacturer badges, and regulatory labels painted on commercial vehicles
# in India. These tokens are filtered out before plate candidate generation to avoid false positives.
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
