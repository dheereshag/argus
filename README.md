# Argus - Automatic Number Plate Recognition (ANPR) API

FastAPI application powered by **FastANPR** for vehicle and truck license plate detection and text recognition.

---

## Features

- **FastANPR Engine**: High-performance ANPR pipeline for license plate localization and character recognition.
- **Base64 JSON API (`POST /recognise`)**: Accepts base64-encoded image payloads and returns exact detection and recognition attributes.
- **Multipart Upload API (`POST /detect`)**: Direct image file upload support.

---

## How to Run with Hot Reload

```bash
uv run fastapi dev
```

Or using Uvicorn directly:

```bash
uv run uvicorn main:app --reload
```

The application runs on `http://127.0.0.1:8000`.

---

## API Documentation & Code Examples

### 1. Python Client Request (`POST /recognise`)

```python
import base64
import requests

# Read image file and convert to base64
image_path = 'tests/images/image001.jpg'
with open(image_path, 'rb') as image_file:
    base64_image_str = base64.b64encode(image_file.read()).decode('utf-8')

# Send POST request
response = requests.post(
    url='http://127.0.0.1:8000/recognise',
    json={'image': base64_image_str}
)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Failed with status code {response.status_code}")
```

#### Sample Response:

```json
{
  "number_plates": [
    {
      "det_box": [682, 414, 779, 455],
      "det_conf": 0.29964497685432434,
      "rec_poly": [[688, 420], [775, 420], [775, 451], [688, 451]],
      "rec_text": "BVH826",
      "rec_conf": 0.940690815448761
    }
  ]
}
```

---

### 2. Direct Library Usage (`FastANPR`)

```python
import cv2
from fastanpr import FastANPR

# Create FastANPR instance
fast_anpr = FastANPR()

# Load images as BGR -> RGB numpy ndarrays
files = ['tests/images/image001.jpg']
images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]

# Run ANPR asynchronously
number_plates = await fast_anpr.run(images)

for file, plates in zip(files, number_plates):
    print(file)
    for plate in plates:
        print("Detection bounding box:", plate.det_box)
        print("Detection confidence:", plate.det_conf)
        print("Recognition text:", plate.rec_text)
        print("Recognition polygon:", plate.rec_poly)
        print("Recognition confidence:", plate.rec_conf)
```

---

### 3. Testing via Swagger UI & `curl`

- **Interactive Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **Multipart Upload (`POST /detect`)**:
  ```bash
  curl -X 'POST' \
    'http://127.0.0.1:8000/detect' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@tests/images/image001.jpg'
  ```
