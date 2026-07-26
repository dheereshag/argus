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

# Regex matching Indian License Plates (Standard: MH01AA2345, RJ09GA0165, MH.01.AA.2345 or BH series: 22BH1234AA)
# Requires valid state prefix, 2-digit RTO code, 1-3 letter series, and exactly 4-digit serial number.
INDIAN_PLATE_REGEX = re.compile(
    r'(?:'
    rf'({STATE_PREFIX_PATTERN})[\s.-]?(\d{{2}})[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{4}})'  # Standard RTO format (State + 2 digits + 1-3 letters + 4 digits)
    r'|'
    r'(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})'                            # Bharat (BH) series
    r')',
    re.IGNORECASE
)
