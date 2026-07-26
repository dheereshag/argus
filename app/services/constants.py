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
    "BP": "Bharat / Police / Custom Series",
}

STATE_PREFIX_PATTERN = "|".join(sorted([k for k in STATE_CODES.keys() if k != "BH"], key=len, reverse=True))

# Regex matching Indian License Plates.
# Standard format: State(2L) + RTO(1-2 alphanum) + Series(1-3L) + Serial(3-4 digits)
# Examples: MH04BG649, DL1CX2744, PBNFD2345, RJ09GA0165
# BH series: 22BH1234AA
# RTO/District code is 1-2 alphanumeric:
#   - 1-2 digits for most states (e.g. MH04, DL1)
#   - Single letter for some states with letter-based RTO sub-divisions (e.g. PBN = Punjab, N office)
# Serial is 3-4 digits: modern plates use 4 digits (with leading zeros), older/smaller RTOs may use 3.
INDIAN_PLATE_REGEX = re.compile(
    r'(?:'
    rf'({STATE_PREFIX_PATTERN})[\s.-]?([A-Z0-9]{{1,2}})[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{3,4}})'  # Standard: State + 1-2 alphanum district + 1-3 letter series + 3-4 digit serial
    r'|'
    r'(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})'                                        # Bharat (BH) series
    r')',
    re.IGNORECASE
)
