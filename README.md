# Argus ANPR Engine

An Enterprise Automatic Number Plate Recognition (ANPR) Python engine built with **YOLO v11**, **Docling OCR (RapidOCR ONNX Runtime)**, and **Strategy & Factory Design Patterns**.

It features an intelligent **YOLO v11 Pre-screening Pipeline** to verify 4-wheeler vehicle presence (`car`, `bus`, `truck`) before routing to downstream OCR / Vision AI models (**Docling Strategy**, **NVIDIA Llama-3.2-11b-Vision**, or **Plate Recognizer**).

---

## ⚙️ Environment Variables (`.env`)

Set the following environment variables in your local `.env` file:

```env
PLATE_RECOGNIZER_TOKEN=your_plate_recognizer_api_token
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_INVOKE_URL=https://integrate.api.nvidia.com/v1/chat/completions
DEFAULT_PROVIDER=docling
DOCLING_OCR_ENGINE=rapidocr
YOLO_MODEL_NAME=yolo11n.pt
YOLO_CONFIG_DIR=/tmp/Ultralytics
HUMAN_CONF_THRESH=0.30
VEHICLE_CONF_THRESH=0.35
```

---

## 🛠️ Installation & Usage

### 1. Installation
```bash
uv sync
```

### 2. Run License Plate Recognition via CLI
```bash
uv run python -m app.main path/to/image.jpg
```

### 3. Use as a Python Library
```python
from app.services.pipeline import recognize_plate_image

# Process image file or raw bytes
response = recognize_plate_image("path/to/image.jpg")

if response.success:
    for plate in response.results:
        print(f"Plate: {plate.plate}, State: {plate.state}")
else:
    print(f"Failed: {response.status_message}")
```

---

## 🧪 Evaluation & Direct Testing

### Run Direct Strategy Benchmark
```bash
uv run python test_direct.py docling
```

### Run Evaluation Against Ground Truth
```bash
uv run python eval.py --docling
```

