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

### Health Check

`GET /`

```bash
curl -s http://localhost:8000/
```

**Response (`200 OK`):**
```json
{
  "name": "Argus ANPR Engine",
  "version": "0.1.0",
  "status": "running",
  "docs": "/docs"
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

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "rejected": false,
  "status": "success",
  "human_count": 0,
  "filename": "1.jpg",
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

**Policy Rejection Response (`200 OK`):**
```json
{
  "success": false,
  "rejected": true,
  "status": "rejected_human_detected",
  "human_count": 2,
  "filename": "frame.jpg",
  "results": [],
  "execution_time_ms": 14.82
}
```

Rejection statuses include: `rejected_human_detected`, `rejected_multiple_vehicles`, and `rejected_no_four_wheeler`.

---

## Python SDK Usage

Argus can be imported directly into Python without running the HTTP server:

```python
from app.services.pipeline import recognize_plate_image

# Process an image file path or raw bytes
response = recognize_plate_image("path/to/vehicle.jpg")

if response.rejected:
    print(f"Rejected: {response.status.value}")
elif response.success:
    for plate in response.results:
        print(f"Plate: {plate.plate} ({plate.state}), Confidence: {plate.confidence:.2f}")
```

---

## Configuration (`.env`)

Operational thresholds and model settings are configured via environment variables or `.env` (see [`.env.example`](.env.example)):

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `YOLO_MODEL_NAME` | `str` | `yolo26n.pt` | Path or name of YOLO26 model weights. |
| `YOLO_CONFIG_DIR` | `str` | `.cache/ultralytics` | Ultralytics cache directory for model downloads. |
| `HUMAN_CONF_THRESH` | `float` | `0.30` | Minimum confidence threshold for pedestrian detection. |
| `VEHICLE_CONF_THRESH` | `float` | `0.35` | Minimum confidence threshold for 4-wheeler detection. |
| `MAX_ALLOWED_HUMANS` | `int \| null` | `0` | Max external pedestrians permitted (`0` = reject on human, `null` = disabled). |
| `ALLOW_CAB_OCCUPANTS` | `bool` | `true` | Exclude occupants/artwork inside vehicle bounding box from rejection. |
| `MAX_ALLOWED_VEHICLES` | `int \| null` | `1` | Max 4-wheelers allowed on scale (`null` = disabled). |
| `MIN_ALLOWED_VEHICLES` | `int` | `1` | Min 4-wheelers required on scale (`0` = allow close-up crop OCR). |
| `MIN_HUMAN_BOX_AREA_RATIO` | `float` | `0.005` | Minimum bbox area ratio to filter background pedestrian noise. |
| `MIN_VEHICLE_BOX_AREA_RATIO` | `float` | `0.01` | Minimum bbox area ratio to filter distant background vehicles. |
| `DEFAULT_YOLO_IMGSZ` | `int` | `640` | YOLO inference image size. |
| `FALLBACK_OCR_ON_NO_VEHICLE` | `bool` | `true` | Attempt full-frame OCR when YOLO misses vehicle body (half-in-frame / bumper). |
| `MAX_CONCURRENT_INFERENCES` | `int` | `4` | Semaphore concurrency limit for model execution. |
| `ONNX_NUM_THREADS` | `int` | `4` | Intra-op thread count for ONNX Runtime (Cortex-A76 quad-core). |
| `IMAGE_RESAMPLE_FILTER` | `str` | `BILINEAR` | Downsampling filter (`BILINEAR` for fast ARM SIMD, `BICUBIC`, `LANCZOS`). |
| `MAX_UPLOAD_BYTES` | `int` | `8388608` | Max HTTP upload payload size (8 MB). |
| `MAX_IMAGE_EDGE_PX` | `int` | `1920` | Max dimension before automatic downscaling. |
| `MAX_IMAGE_PIXELS` | `int` | `50000000` | Max pixel threshold for decompression bomb protection. |
| `SERVER_HOST` | `str` | `127.0.0.1` | Server bind host interface. |
| `SERVER_PORT` | `int` | `8000` | Server HTTP listening port. |

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
