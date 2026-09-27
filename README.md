# Argus — ANPR Microservice

Automatic Number Plate Recognition (ANPR) microservice designed for weighbridge gatekeeping. Combines **Ultralytics YOLO** for vehicle localization and **RapidOCR (ONNX Runtime)** for Indian license plate reading.

---

## 🚀 How to Run

### 1. Local Development

```bash
# Install dependencies
uv sync

# Run development server (hot-reload)
uv run fastapi dev

# Or run in production mode locally
uv run fastapi run --host 127.0.0.1
```

Interactive documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### 2. Remote Pi Deployment (1-Command)

Compiles `app` with Nuitka into a native C extension (`app.*.so`), syncs model weights (`yolo26n.pt`), purges all `.py` source files from client hardware, and configures systemd (`argus.service`) bound to `127.0.0.1:8000` with auto-restart on boot:

```bash
uv run python scripts/deploy.py --host gluvok@hermes.local
```

---

## 📡 API Reference: `/recognize`

### Request (POST)

Accepts a multipart file upload (`file`, up to 8 MB):

```bash
curl -X POST "http://127.0.0.1:8000/recognize" \
  -H "Accept: application/json" \
  -F "file=@tests/1.jpg"
```

### Response (`200 OK`)

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

- **`plate`**: Cleaned and validated plate number string (or `null` if none detected).
- **`vehicle_type`**: Detected vehicle category (`car`, `truck`, `bus`, `motorcycle`, etc.).
- **`state`**: State of vehicle registration.
- **`confidence`**: OCR detection confidence score (0.0 to 1.0).
- **`box`**: Detected plate bounding box `[ymin, xmin, ymax, xmax]`.
- **`humans_outside` / `humans_inside`**: Occupancy count around the weighbridge.
