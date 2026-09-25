# Argus

Automatic Number Plate Recognition (ANPR) microservice and Python library designed for weighbridge gatekeeping and vehicle access control. Combines **Ultralytics YOLO26** for vehicle localization and human detection with **RapidOCR (ONNX Runtime)** for Indian license plate reading and character disambiguation.

---

## Quick Start

Argus uses **[uv](https://docs.astral.sh/uv/)** for dependency and environment management.

### Installation

```bash
git clone https://github.com/dheereshag/argus.git
cd argus
uv sync
```

### Run Server

```bash
# Development (with hot-reload):
uv run fastapi dev

# Production:
uv run fastapi run --host 127.0.0.1
```

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## REST API Reference

### Service Info & Health Checks

`GET /` (Service Info) or `GET /health` (Liveness / Readiness Probe)

```bash
curl -s http://localhost:8000/health
```

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "service": "Argus ANPR Microservice",
  "version": "0.1.0"
}
```

### Recognize Plate

`POST /recognize`

Accepts a multipart file upload (`file`, up to 8 MB) and returns detection and OCR results.

```bash
curl -X POST "http://localhost:8000/recognize" \
  -H "Accept: application/json" \
  -F "file=@tests/1.jpg"
```

**Response (`200 OK`):**
```json
{
  "filename": "1.jpg",
  "humans_outside": 0,
  "humans_inside": 1,
  "results": [
    {
      "plate": "RJ09GA0165",
      "vehicle_type": "car",
      "state": "Rajasthan",
      "raw_text": "RJ09 GA 0165",
      "confidence": 0.98,
      "box": [412, 530, 624, 592]
    }
  ],
  "execution_time_ms": 43.15
}
```

- `humans_outside`: Count of pedestrians detected outside vehicle bounds.
- `humans_inside`: Count of human occupants detected inside vehicle cabins.
- `results`: ANPR detection results. When vehicles are detected, per-vehicle crop OCR extracts plates directly attached to each vehicle (`car`, `truck`, `bus`, `motorcycle`, `bicycle`). If any vehicle has a recognized plate and `ENABLE_FULL_FRAME_OCR=false`, full-frame OCR is skipped. If vehicles are detected but none yield a readable plate, full-frame OCR automatically executes as a fallback. When `ENABLE_FULL_FRAME_OCR=true`, secondary full-frame OCR runs unconditionally to capture and spatially associate unlocalized plates. By default (`INCLUDE_UNIDENTIFIED_VEHICLES=false`), `results` contains only entries with detected license plates; when `INCLUDE_UNIDENTIFIED_VEHICLES=true`, unplated vehicles are preserved with `plate: null`. When zero vehicles are detected, full-frame fallback returns all valid plates with `vehicle_type: null`; if no plate is found, `results` is `[]`.

---

## Python SDK Usage

Argus can be imported directly into Python without running the HTTP server:

```python
from app.services.pipeline import recognize_plate_image

# Process an image file path or raw bytes
response = recognize_plate_image("path/to/vehicle.jpg")

print(f"Pedestrians: {response.humans_outside}, Cab Occupants: {response.humans_inside}")
for res in response.results:
    if res.plate:
        print(f"Plate: {res.plate} ({res.state}), Vehicle: {res.vehicle_type}, Confidence: {res.confidence:.2f}")
    else:
        print(f"Vehicle: {res.vehicle_type}, Plate: Not detected")
```

---

## Configuration (`.env`)

The service exposes two operational pipeline booleans configurable via environment variables or `.env` (see [`.env.example`](.env.example)):

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENABLE_FULL_FRAME_OCR` | `bool` | `false` | Enable secondary full-frame OCR pass when vehicles are detected. If false, full-frame OCR only runs as fallback when zero plates are recognized from crops. |
| `INCLUDE_UNIDENTIFIED_VEHICLES` | `bool` | `false` | If false, `results` only contains entries with detected license plates. If true, vehicles without recognized plates are included as `plate: null`. |

All other operational parameters (model weights, detection confidence thresholds, image upload limits, NMS thresholds, threadpool counts) are fixed internal constants defined in [`app/core/config.py`](app/core/config.py).

---

## Indian License Plate Rules

- **Standard State Series**: `SS DD AA NNNN` or `SS DD AAA NNNN` (e.g. `MH12AB1234`, `DL01CAA1234`, `RJ09GA0165`).
- **Bharat Series (BH)**: `YY BH NNNN AA` (e.g. `22BH1234AA`).
- **Government / Police**: `BP` departmental series.
- **Positional OCR Disambiguation**:
  - Alphabet positions (State / Series): `0 -> O`, `1 -> I`, `2 -> Z`, `4 -> A`, `5 -> S`, `8 -> B`.
  - Numeric positions (District / Number): `O -> 0`, `D -> 0`, `I -> 1`, `L -> 1`, `Z -> 2`, `B -> 8`.
  - State prefix misreads: `W8 -> WB`, `7G -> TG`, `RT -> RJ`, `D1 -> DL`, `0D -> OD`.
- **HSRP Band & Decals**: Automatically strips fused `"IND"` prefixes and filters common truck decal text (`GOODS CARRIER`, `TATA`, 10-digit mobile numbers).
- Supported state codes are defined in [`app/constants.py`](app/constants.py).

---

## Verification & Quality Gates

All modifications must pass the three verification gates defined in [`AGENTS.md`](AGENTS.md):

```bash
# 1. Lint & style check
uv run ruff check --fix

# 2. Static type checking
uv run ty check

# 3. Test suite
uv run pytest
```

---

## Documentation

- **[Architecture & Pipeline Guide](docs/ARCHITECTURE.md)**: Detailed breakdown of the two-stage pipeline, spatial clustering, data contracts, and concurrency model.
- **[Raspberry Pi 5 Optimization Guide](docs/RASPBERRY_PI_5_OPTIMIZATION.md)**: Hardware tuning, Python 3.14 free-threading, ONNX thread affinity, and Hailo-8 NPU acceleration.
- **[Edge Security & Hardening Guide](docs/EDGE_SECURITY.md)**: Deployment guidelines for Raspberry Pi edge devices, Nuitka compilation, and physical security.
