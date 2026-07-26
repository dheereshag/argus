# Argus ANPR Microservice

An Enterprise Automatic Number Plate Recognition (ANPR) microservice built with **FastAPI**, **YOLO v11**, and the **Strategy & Factory Design Patterns**.

It features an intelligent **YOLO v11 Pre-screening Pipeline** to verify 4-wheeler vehicle presence (`car`, `bus`, `truck`) and enforce zero human occupancy before routing to downstream OCR / Vision AI models (**Plate Recognizer** or **NVIDIA Llama-3.2-11b-Vision**).

---

## 🚀 Deployment on Render

### Start Command (Production)
```bash
uv run uvicorn main:app --host 0.0.0.0 --port $PORT
```
> Render automatically injects the `$PORT` environment variable.

### Build Command
```bash
uv sync
```

---

## ⚙️ Environment Variables (`.env`)

Set the following environment variables on your Render Dashboard or in a local `.env` file:

```env
PLATE_RECOGNIZER_TOKEN=your_plate_recognizer_api_token
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_INVOKE_URL=https://integrate.api.nvidia.com/v1/chat/completions
DEFAULT_PROVIDER=platerecognizer
YOLO_MODEL_NAME=yolo11n.pt
HUMAN_CONF_THRESH=0.30
VEHICLE_CONF_THRESH=0.35
```

---

## 🛠️ Local Development

### Installation & Sync
```bash
uv sync
```

### Run Dev Server
```bash
uv run fastapi dev
```
Interactive OpenAPI Documentation: **`http://localhost:8000/docs`**

---

## 📡 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Available Providers
`GET /providers`
```json
{
  "available_providers": ["platerecognizer", "nvidia"],
  "default_provider": "platerecognizer"
}
```

### 3. License Plate Recognition
`POST /recognize`

**Query Parameters:**
- `provider` *(optional)*: `"platerecognizer"` | `"nvidia"`. Defaults to `DEFAULT_PROVIDER`.

**Form Data:**
- `file`: Vehicle image file (`JPEG`/`PNG`).

#### Example Response (Success)
```json
{
  "success": true,
  "status": "success",
  "status_message": "License plate successfully detected and recognized on truck.",
  "vehicle_detected": true,
  "vehicle_type": "truck",
  "human_detected": false,
  "filename": "car.jpg",
  "provider": "platerecognizer",
  "results": [
    {
      "plate": "RJ14GT4976",
      "state": "Rajasthan"
    }
  ],
  "execution_time_ms": 1450.23
}
```

#### Example Response (Early Rejection - Human Detected)
```json
{
  "success": false,
  "status": "rejected_human_detected",
  "status_message": "Image rejected: Human presence detected.",
  "vehicle_detected": true,
  "vehicle_type": "truck",
  "human_detected": true,
  "filename": "car2.jpg",
  "provider": "platerecognizer",
  "results": [],
  "execution_time_ms": 85.12
}
```

---

## 🏛️ Status Enums (`RecognitionStatusEnum`)

| Status Enum | Description |
| :--- | :--- |
| `success` | 4-wheeler detected, no human occupancy, plate recognized |
| `rejected_no_four_wheeler` | No 4-wheeler vehicle (`car`, `bus`, `truck`) detected |
| `rejected_human_detected` | Human presence detected in frame |
| `no_plate_detected` | Passed pre-screening, but plate OCR failed |
