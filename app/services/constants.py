import re

# Indian State & Union Territory Codes Mapping
STATE_CODES = {
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
    "BH": "Bharat Series (National)",
}

STATE_PREFIX_PATTERN = "|".join(sorted([k for k in STATE_CODES.keys() if k != "BH"], key=len, reverse=True))

# Regex matching Indian License Plates.
# Standard format: State(2L) + District(2 digits 01-99, or 1 digit for DL) + Series(1-3L) + Serial(3-4 digits)
# BH series: Year(2 digits) + BH + Serial(4 digits) + Series(1-2L)
# Examples: MH04BG649, DL1CX2744, RJ09GA0165, RJ14GJ4976, 22BH1234AA
INDIAN_PLATE_REGEX = re.compile(
    r'(?:'
    rf'({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d|(?<=DL)[1-9])[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{3,4}})'
    r'|'
    r'(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})'
    r')',
    re.IGNORECASE
)
