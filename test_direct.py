import os
import re
import requests
from pprint import pprint

API_TOKEN = "422624c83642090213bf7922c0426d762be64215"
TESTS_DIR = "tests"
REGIONS = ["in"]  # India country code for Plate Recognizer API

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

# Regex for Indian License Plates (Standard: MH1AA2345, RJ09GA0165, MH.1AA.2345 or BH series: 22BH1234AA)
INDIAN_PLATE_REGEX = re.compile(
    r'^(?:'
    r'([A-Za-z]{2})[\s.-]?(\d{1,2}[A-Za-z]?)[\s.-]?([A-Za-z]{1,3})[\s.-]?(\d{1,4})'  # Standard & New RTO format
    r'|'
    r'(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})'                      # Bharat (BH) series
    r')$',
    re.IGNORECASE
)

def extract_number_plates(api_response):
    """
    Extracts and returns details (plate number and state name) for valid Indian number plates
    from Plate Recognizer API response.
    """
    results = api_response.get("results", [])
    output = []

    for res in results:
        candidates = [res.get("plate", "")] + [c.get("plate", "") for c in res.get("candidates", [])]
        
        matched_plate = None
        state_name = None
        
        for cand in candidates:
            if not cand:
                continue
            cand_clean = cand.strip().upper()
            match = INDIAN_PLATE_REGEX.match(cand_clean)
            if match:
                matched_plate = cand_clean
                # Extract state code (Group 1 for standard format, Group 6 for BH series)
                state_code = match.group(1) or match.group(6)
                if state_code:
                    state_name = STATE_CODES.get(state_code.upper(), "Unknown State")
                break

        if matched_plate:
            output.append({
                "plate": matched_plate,
                "state": state_name
            })
        elif res.get("plate"):
            raw_plate = res.get("plate").upper()
            state_code = raw_plate[:2]
            output.append({
                "plate": raw_plate,
                "state": STATE_CODES.get(state_code, "Unknown State")
            })

    return output

def test_plate_recognizer():
    images = [f for f in os.listdir(TESTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    for img_name in images:
        img_path = os.path.join(TESTS_DIR, img_name)
        print(f"\n{'='*50}\nTesting image: {img_path}\n{'='*50}")
        
        with open(img_path, 'rb') as fp:
            response = requests.post(
                'https://api.platerecognizer.com/v1/plate-reader/',
                data=dict(regions=REGIONS),
                files=dict(upload=fp),
                headers={'Authorization': f'Token {API_TOKEN}'}
            )
        
        if response.status_code in (200, 201):
            data = response.json()
            plates_info = extract_number_plates(data)
            print("Extracted Info:", plates_info)
        else:
            print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_plate_recognizer()


