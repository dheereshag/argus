# Argus ANPR Engine

An Enterprise Automatic Number Plate Recognition (ANPR) Python engine built with **YOLO v11** and **Docling OCR (RapidOCR ONNX Runtime)**.

It features an intelligent **YOLO v11 Pre-screening Pipeline** to verify 4-wheeler vehicle presence (`car`, `bus`, `truck`) and occupancy before executing downstream **Docling OCR** plate extraction and Indian license plate regex validation.

---

## ⚙️ Environment Variables (`.env`)

Set the following environment variables in your local `.env` file (see [`.env.example`](file:///.env.example)):

```env
# YOLO Model Settings
YOLO_MODEL_NAME=yolo11n.pt
YOLO_CONFIG_DIR=/tmp/Ultralytics
HUMAN_CONF_THRESH=0.30
VEHICLE_CONF_THRESH=0.35

# Pre-screening Rejection Policies
REJECT_ON_HUMAN_DETECTED=false
REJECT_ON_MULTIPLE_VEHICLES=false
REJECT_ON_NO_VEHICLE=false

# Processing & Upload Limits
MAX_OCR_LINES=500
MAX_UPLOAD_BYTES=8388608
MAX_IMAGE_PIXELS=50000000
MAX_IMAGE_EDGE_PX=1920

# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

---

## 🛠️ Installation & Usage

### 1. Installation
```bash
uv sync
```

### 2. Start FastAPI REST API Server
```bash
uv run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Requests:

- **Root Info**:
  ```bash
  curl http://localhost:8000/
  ```

- **Recognize License Plate from Image**:
  ```bash
  curl -X POST "http://localhost:8000/recognize" \
    -F "file=@path/to/vehicle.jpg"
  ```

### 3. Run License Plate Recognition via CLI
```bash
uv run python -m app.main path/to/image.jpg
```
or:
```bash
uv run python main.py path/to/image.jpg
```

### 4. Use as a Python Library
```python
from app.services.pipeline import recognize_plate_image

# Process image file path or raw bytes
response = recognize_plate_image("path/to/image.jpg")

if response.success:
    for plate in response.results:
        print(f"Plate: {plate.plate}, State: {plate.state}")
else:
    print(f"Failed: {response.status_message}")
```

---

## 🧪 Direct Testing

### Run Direct Pipeline Benchmark
```bash
uv run python test_direct.py
```

---

## 🛡️ Quality & Verification Gates

Per [AGENTS.md](file:///AGENTS.md), all modifications must pass the 3 mandatory gates:

```bash
# 1. Lint & Code Style
uv run ruff check --fix

# 2. Type Checking
uv run ty check

# 3. Test Suite
uv run pytest
```
