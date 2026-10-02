"""Indian state/UT codes and official vehicle registration regex patterns."""

import re

STATE_CODES: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli", "GA": "Goa",
    "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand",
    "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
    "OR": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "TG": "Telangana", "UK": "Uttarakhand", "UA": "Uttarakhand", "UP": "Uttar Pradesh",
    "WB": "West Bengal", "BP": "Police / Government Series",
}

STATE_PREFIX_PATTERN = "|".join(sorted(STATE_CODES.keys(), key=len, reverse=True))

INDIAN_PLATE_REGEX: re.Pattern[str] = re.compile(
    r"(?:"
    rf"({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d|[1-9])[\s.-]?([A-HJ-NP-Za-hj-np-z]{{1,3}})[\s.-]?(\d{{4}})"
    r"|"
    rf"({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d)[\s.-]?(\d{{4}})"
    r")",
    re.IGNORECASE,
)
