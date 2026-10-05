# Argus — ANPR Microservice

Automatic Number Plate Recognition (ANPR) microservice designed for weighbridge gatekeeping. Features a **4-tier cascaded execution pipeline** combining **Fast-ALPR (YOLOv9-s + CCT-XS OCR with margin expansion)** as the primary engine with automatic fallback to **Argus's RapidOCR pipeline**, delivering thin, high-speed response schemas.

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

Compiles `app` with Nuitka into a native C extension (`app.*.so`), syncs model weights (`yolo26n.onnx` / `yolo26n.pt`), purges all `.py` source files from client hardware, and configures systemd (`argus.service`) bound to `127.0.0.1:8000` with auto-restart on boot:

```bash
# Optional: re-export YOLO26 PyTorch weights to ONNX for Pi CPU acceleration
uv run python scripts/export_yolo_onnx.py

# Deploy to Raspberry Pi 5
uv run python scripts/deploy.py --host gluvok@hermes.local
```

---

## ⚡ 4-Tier Cascaded Execution Pipeline

1. **Tier 1 (Fast-ALPR Full-Frame)**: Direct YOLOv9-s plate detector + CCT-XS OCR with margin expansion on input image, normalized & validated against Indian MoRTH rules (`~30–40 ms`). Early exit if valid plate found.
2. **Tier 2 (Fast-ALPR on Vehicle Crops)**: If Tier 1 produces no plates, YOLO26 vehicle detector (ONNX Runtime CPU via `yolo26n.onnx`) extracts vehicle crops $\rightarrow$ Fast-ALPR plate detection + OCR on crops (`~50–70 ms`). Early exit if valid plate found.
3. **Tier 3 (RapidOCR on Vehicle Crops)**: If Tier 2 produces no plates, executes RapidOCR 2D spatial candidate pairing on vehicle crops (`~120–180 ms`). Early exit if valid plate found.
4. **Tier 4 (RapidOCR Full-Frame Fallback)**: If Tier 3 produces no plates or no vehicle was detected, runs RapidOCR across the full frame (`~220–280 ms`).

---

## 📡 API Reference: `/recognize`

### Request (POST)

Accepts a multipart file upload (`file`, up to 8 MB):

```bash
curl -X POST "http://127.0.0.1:8000/recognize" \
  -H "Accept: application/json" \
  -F "file=@tests/images/1.jpg"
```

### Response (`200 OK`)

```json
{
  "results": [
    {
      "plate": "RJ09GA0165",
      "execution_time_ms": 31.42
    }
  ],
  "execution_time_ms": 31.42
}
```

- **`results`**: List of validated plate items, each with `plate` registration number and item `execution_time_ms`.
- **`execution_time_ms`**: Total end-to-end processing latency in milliseconds.
