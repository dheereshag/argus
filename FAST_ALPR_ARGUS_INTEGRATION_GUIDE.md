# Fast-ALPR Primary Engine & 4-Tier Fallback Integration Guide for Argus

This technical specification and implementation guide is designed to be provided directly to an AI coding assistant (or software engineer) to integrate **Fast-ALPR** as the **primary ANPR engine** with a **4-tier cascaded fallback architecture** into Argus, while slimming down the API response to a **thin schema** and **removing human occupancy counting**.

---

## 1. Requirements & Core Objectives

1. **4-Tier Cascaded Execution Hierarchy**:
   - **Tier 1 (Fast-ALPR Full-Frame)**: Run Fast-ALPR directly on the input image (YOLOv9-s plate detector + CCT-XS OCR with margin expansion). Candidates are ordered by confidence descending, then validated and normalized against Indian MoRTH rules via `parse_plate_info()`. If valid plates are detected, return immediately (~30–40 ms).
   - **Tier 2 (Fast-ALPR on Vehicle Crops)**: If Tier 1 yields no valid plates, run YOLO26 vehicle detection to isolate vehicle crop(s). Execute Fast-ALPR plate detection + CCT-XS OCR on each vehicle crop, followed by `parse_plate_info()`. If valid plates are detected, return immediately (~50–70 ms).
   - **Tier 3 (Argus RapidOCR on Vehicle Crops)**: If Tier 2 yields no valid plates, execute Argus's RapidOCR pipeline on the cropped vehicle boxes (2D spatial candidate pairing, unsharp/black-hat enhancement, and regex validation). If valid plates are detected, return immediately (~120–180 ms).
   - **Tier 4 (Argus RapidOCR Full-Frame Fallback)**: If Tier 3 yields no valid plates (or if no vehicles were detected by YOLO26), run Argus's RapidOCR engine on the entire full frame. If valid plates are found, return them. If still empty, return an empty results array (~220–280 ms).
2. **Indian Plate Validation & Normalization**:
   - All recognized plates across every tier are passed through Argus's `parse_plate_info()` to:
     - Strip HSRP hologram prefixes (`IND`, `1ND`, etc.).
     - Normalize state code typos (`RT -> RJ`, `W8 -> WB`, etc.).
     - Validate against MoRTH standard regex and Bharat Series (`BH`) format.
3. **Thin API Response**:
   - Eliminate heavy metadata: remove `plates`, `vehicle_type`, `state`, `raw_text`, `confidence`, `box`, and `filename`.
   - The response contains only `results` (with `plate`), and `execution_time_ms`.
4. **Remove Human Tracking**:
   - Remove `humans_inside` and `humans_outside` from schemas, detectors, and pipelines.
5. **NASA JPL Rule 4 Invariant**:
   - **Every single `.py` file must be $\le$ 60 lines.**
   - **Every single function must be $\le$ 60 lines.**

---

## 2. Target Pipeline Architecture

```
                               POST /recognize (Image Input)
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │      Tier 1: Fast-ALPR (Full-Frame)       │
                      │ - Direct YOLOv9-s Plate Detector          │
                      │ - Margin Padding (15% X, 10% Y)           │
                      │ - CCT-XS OCR                              │
                      │ - Indian MoRTH Validation & Normalization │
                      └─────────────────────┬─────────────────────┘
                                            │
                                  Valid plate(s) found?
                                          /   \
                                   YES   /     \   NO (empty)
                                        /       \
                                       v         ▼
                     ┌───────────────────┐  ┌───────────────────────────┐
                     │                   │  │ Tier 2: Vehicle Detection │
                     │                   │  │ - YOLO26 Vehicle Detector │
                     │                   │  │ - Extract Vehicle Crop(s) │
                     │                   │  └─────────────┬─────────────┘
                     │                   │                │
                     │                   │                ▼
                     │                   │  ┌───────────────────────────┐
                     │                   │  │ Tier 2: Fast-ALPR on Crop │
                     │                   │  │ - YOLOv9-s on Crop        │
                     │                   │  │ - Indian MoRTH Validation │
                     │                   │  └─────────────┬─────────────┘
                     │                   │                │
                     │                   │      Valid plate(s) found?
                     │                   │              /   \
                     │                   │       YES   /     \   NO
                     │                   │            /       \
                     │  Thin Response    │           v         ▼
                     │  Construction     │ ┌───────────────────┐  ┌───────────────────────────┐
                     │  & Return         │ │                   │  │ Tier 3: RapidOCR on Crop  │
                     │  (Early Exit)     │ │                   │  │ - Spatial 2D Pairing      │
                     │                   │ │                   │  │ - Indian MoRTH Validation │
                     │                   │ │                   │  └─────────────┬─────────────┘
                     │                   │ │                   │                │
                     │                   │ │                   │      Valid plate(s) found?
                     │                   │ │                   │              /   \
                     │                   │ │                   │       YES   /     \   NO
                     │                   │ │                   │            /       \
                     │                   │ │                   │           v         ▼
                     │                   │ │                   │ ┌───────────────────┐  ┌───────────────────────────┐
                     │                   │ │                   │ │                   │  │ Tier 4: RapidOCR Full-    │
                     │                   │ │                   │ │                   │  │         Frame Fallback    │
                     │                   │ │                   │ │                   │  └─────────────┬─────────────┘
                     │                   │ │                   │ │                   │                │
                     └───────────────────┘ └───────────────────┘ └───────────────────┘                │
                               ▲                     ▲                     ▲                          │
                               │                     │                     │                          │
                               └─────────────────────┴─────────────────────┴──────────────────────────┘
                                                              │
                                                              ▼
                                              ┌───────────────────────────────┐
                                              │      Thin JSON Response       │
                                              │ {                             │
                                              │   "results": [                │
                                              │     {"plate": "RJ09GA0165"}   │
                                              │   ],                          │
                                              │   "execution_time_ms": 31.42  │
                                              │ }                             │
                                              └───────────────────────────────┘
```

---

## 3. Thin Response Schema Specification

In `app/schemas.py`:

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

- **`results`**: List of thin objects containing `"plate"` and per-result `"execution_time_ms"`.
- **`execution_time_ms`**: Total roundtrip pipeline duration.
- **Removed**: `plates`, `filename`, `humans_inside`, `humans_outside`, `vehicle_type`, `state`, `raw_text`, `confidence`, `box`.

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Update Dependencies in `pyproject.toml`
In `../repos/argus/pyproject.toml`, add `fast-alpr[onnx]>=0.4.0` to `dependencies`:

```toml
dependencies = [
    "asyncer>=0.0.18",
    "fast-alpr[onnx]>=0.4.0",
    "fastapi[standard]>=0.141.1",
    "loguru>=0.7.2",
    "numpy>=2.5.1",
    "onnxruntime>=1.29.0",
    "opencv-python-headless>=5.0.0.93",
    "pillow>=12.3.0",
    "pydantic>=2.13.4",
    "pydantic-settings>=2.15.0",
    "rapidocr>=3.9.2",
    "ultralytics>=8.4.106",
]
```
Run `uv sync` in `repos/argus`.

---

### Step 2: Simplify `app/schemas.py` (Thin Schema & Remove Humans)
Modify `app/schemas.py` to remove human counts and make `PlateResult` and `RecognitionResponse` thin:

```python
"""Domain models, internal dataclasses, and thin API response schemas for Argus ANPR."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True, frozen=True)
class OCRToken:
    text: str
    score: float
    cx: float | None = None
    cy: float | None = None
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class PlateCandidate:
    y_pos: float
    rank: int
    info: dict[str, Any]
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectedVehicle:
    vehicle_type: str
    box: tuple[int, int, int, int]
    crop: Any = None
    crop_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectionResult:
    vehicles: list[DetectedVehicle] = field(default_factory=list)


class PlateResult(BaseModel):
    """Thin plate result containing only the validated registration number."""

    plate: str = Field(description="Normalized Indian vehicle registration number")


class RecognitionResponse(BaseModel):
    """Thin API response schema for license plate recognition."""

    results: list[PlateResult] = Field(default_factory=list, description="Plate result items")
    execution_time_ms: float = Field(..., description="Processing duration in milliseconds")


class APIErrorResponse(BaseModel):
    success: bool = Field(False)
    status_code: int = Field(...)
    message: str = Field(...)
    error_type: str = Field(...)
    details: Any = Field(None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 3: Create Fast-ALPR Engine Singleton
Create file `app/services/ocr/fast_alpr_engine.py`:

```python
"""Fast-ALPR Model Singleton and Warmup Engine."""

import threading
from typing import Any

from app.core.logging import logger

_alpr_instance: Any = None
_lock = threading.Lock()


def get_fast_alpr_engine() -> Any:
    """Return thread-safe singleton instance of Fast-ALPR engine."""
    global _alpr_instance
    if _alpr_instance is None:
        with _lock:
            if _alpr_instance is None:
                from fast_alpr import ALPR

                logger.info("Initializing Fast-ALPR ONNX models...")
                _alpr_instance = ALPR(
                    detector_model="yolo-v9-s-608-license-plate-end2end",
                    detector_conf_thresh=0.25,
                    ocr_model="cct-xs-v2-global-model",
                )
    return _alpr_instance


def check_fast_alpr_engine() -> bool:
    """Pre-warm and verify Fast-ALPR models during lifespan startup."""
    try:
        engine = get_fast_alpr_engine()
        return engine is not None
    except Exception as exc:
        logger.error(f"Fast-ALPR warmup failed: {exc}")
        return False
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 4: Create Fast-ALPR Margin Expansion Helper
Create file `app/services/ocr/fast_alpr_padding.py`:

```python
"""Margin expansion utility to prevent edge character truncation."""

from typing import Any

import numpy as np


def compute_padded_crop(
    img: np.ndarray,
    bbox: Any,
    pad_x_pct: float = 0.15,
    pad_y_pct: float = 0.10,
) -> np.ndarray:
    """Expand plate bounding box with margin padding to capture edge characters."""
    h, w = img.shape[:2]
    box_w = bbox.x2 - bbox.x1
    box_h = bbox.y2 - bbox.y1

    px = max(10, int(box_w * pad_x_pct))
    py = max(5, int(box_h * pad_y_pct))

    x1 = max(0, bbox.x1 - px)
    y1 = max(0, bbox.y1 - py)
    x2 = min(w, bbox.x2 + px)
    y2 = min(h, bbox.y2 + py)

    return img[y1:y2, x1:x2]
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 5: Create Fast-ALPR Runner for Full-Frames and Vehicle Crops
Create file `app/services/ocr/fast_alpr_runner.py`:
Supports both **Tier 1** (full-frame) and **Tier 2** (vehicle crops) with confidence sorting and Indian MoRTH validation:

```python
"""Execute Fast-ALPR on full frames and vehicle crops with Indian plate validation."""

import statistics
from typing import Any

import cv2
import numpy as np

from app.services.image_processing import ImageInput, load_rgb
from app.services.ocr.fast_alpr_engine import get_fast_alpr_engine
from app.services.ocr.fast_alpr_padding import compute_padded_crop
from app.services.plate_rules import parse_plate_info


def _calc_conf(ocr_res: Any) -> float:
    if not ocr_res or not ocr_res.confidence:
        return 0.0
    if isinstance(ocr_res.confidence, list):
        return statistics.mean(ocr_res.confidence) if ocr_res.confidence else 0.0
    return float(ocr_res.confidence)


def run_fast_alpr(img_bgr: np.ndarray) -> list[str]:
    """Run Fast-ALPR on a BGR image/crop, sort by confidence, validate Indian plates."""
    alpr = get_fast_alpr_engine()
    detections = alpr.detector.predict(img_bgr)
    if not detections:
        return []
    candidates: list[tuple[float, str]] = []
    for det in detections:
        crop = compute_padded_crop(img_bgr, det.bounding_box)
        ocr_res = alpr.ocr.predict(crop)
        if ocr_res and ocr_res.text:
            candidates.append((_calc_conf(ocr_res), ocr_res.text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    valid_plates: list[str] = []
    for _, raw_text in candidates:
        if (p_info := parse_plate_info(raw_text)) and (p := p_info.get("plate")):
            if p not in valid_plates:
                valid_plates.append(p)
    return valid_plates


def run_fast_alpr_pipeline(image_input: ImageInput) -> list[str]:
    """Tier 1: Run Fast-ALPR directly on full input image."""
    pil_img = load_rgb(image_input)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return run_fast_alpr(img_bgr)


def run_fast_alpr_on_crops(vehicles: list[Any]) -> list[str]:
    """Tier 2: Run Fast-ALPR on cropped vehicle boxes."""
    for v in vehicles:
        if v.crop is None:
            continue
        crop_bgr = cv2.cvtColor(np.array(v.crop), cv2.COLOR_RGB2BGR)
        if plates := run_fast_alpr(crop_bgr):
            return plates
    return []
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 6: Update Detector to Remove Human Partitioning
In `app/services/detector/detector.py`:
- Remove calls to `partition_humans`.
- Only collect vehicle detections.
- Simplify `detect()` to return `DetectionResult(vehicles=...)`.

```python
# app/services/detector/detector.py
import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core import constants
from app.core.contracts import ensure, require
from app.schemas import DetectedVehicle, DetectionResult
from app.services.detector.geometry import BoundingBox, clamp_box, pad_box
from app.services.detector.parser import parse_detections
from app.services.image_processing import ImageInput, load_rgb


class VehicleDetector:
    """Stage 1: YOLO26 Vehicle Detection."""

    _model: YOLO | None = None
    _clamp_box, _pad_box = staticmethod(clamp_box), staticmethod(pad_box)
    _parse_detections = staticmethod(parse_detections)

    @classmethod
    def get_model(cls) -> YOLO:
        if cls._model is None:
            import app.services.detector as yf

            cls._model = yf.YOLO(constants.YOLO_MODEL_NAME or "yolo26n.pt")
        ensure(cls._model is not None, "YOLO model failed to initialise")
        return cls._model

    def _run_detection(self, img: Image.Image, v_conf: float) -> list[tuple[str, BoundingBox]]:
        require(img is not None, "_run_detection called with no image")
        res = next(iter(self.get_model()(img, imgsz=constants.DEFAULT_YOLO_IMGSZ, agnostic_nms=constants.YOLO_AGNOSTIC_NMS, verbose=False)))
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0 or not hasattr(boxes, "cls"):
            return []
        c_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        xyxy = (boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)) if getattr(boxes, "xyxy", None) is not None else None
        _, vehicles = parse_detections(c_ids, confs, xyxy, img.width, img.height, 1.0, v_conf)
        return vehicles

    def detect(self, image_input: ImageInput) -> DetectionResult:
        pil_img = load_rgb(image_input)
        vehicles = self._run_detection(pil_img, constants.VEHICLE_CONF_THRESH)
        w, h = pil_img.width, pil_img.height
        detected = [DetectedVehicle(vt, b, pil_img.crop(pad_box(b, w, h)), pad_box(b, w, h)) for vt, b in vehicles]
        return DetectionResult(vehicles=detected)
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 7: Update `app/services/pipeline/response.py` for Thin Schema
Update `app/services/pipeline/response.py`:

```python
"""Response construction for thin ANPR results."""

import time

from app.schemas import PlateResult, RecognitionResponse


def _build_response(
    plates: list[str],
    start_time: float,
) -> RecognitionResponse:
    """Assemble thin RecognitionResponse containing only validated plate results and latency."""
    results = [PlateResult(plate=p) for p in plates]
    return RecognitionResponse(
        results=results,
        execution_time_ms=round((time.time() - start_time) * 1000, 2),
    )
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 8: Update Orchestrator: 4-Tier Cascaded Execution
In `app/services/pipeline/orchestrator.py`:
Implement the 4-tier cascaded fallback with early return at the first successful tier:
1. **Tier 1**: `run_fast_alpr_pipeline(prepared_img)` on full frame.
2. **Tier 2**: `run_fast_alpr_on_crops(detection.vehicles)` on vehicle crops.
3. **Tier 3 & 4**: `_run_rapidocr_fallback(detection, prepared_img, filename)` on vehicle crops then full frame.

```python
"""ANPR Pipeline Orchestrator: 4-tier cascaded fallback with thin response."""

import time
from typing import Any

from app.core.logging import logger
from app.schemas import DetectionResult, RecognitionResponse
from app.services.image_processing import decode_image
from app.services.ocr.fast_alpr_runner import run_fast_alpr_on_crops, run_fast_alpr_pipeline
from app.services.pipeline.helpers import _resolve_bytes
from app.services.pipeline.response import _build_response
from app.services.pipeline.stages import _run_stage2_ocr


def _run_rapidocr_fallback(det: DetectionResult, img: Any, fn: str) -> list[str]:
    """Tier 3 & 4: Argus RapidOCR on vehicle crops with full-frame fallback."""
    logger.info(f"Fast-ALPR produced no plates. Falling back to RapidOCR for '{fn}'")
    fallback_results = _run_stage2_ocr(det, img, fn)
    return [r.plate for r in fallback_results if r.plate and r.plate != "N/A"]


def _execute_cascade(prepared_img: Any, filename: str) -> list[str]:
    """Execute 4-tier cascade: Fast-ALPR full -> Fast-ALPR crop -> RapidOCR crop -> RapidOCR full."""
    # Tier 1: Fast-ALPR directly on full frame
    if plates := run_fast_alpr_pipeline(prepared_img):
        return plates

    # Tier 2: Vehicle detection + Fast-ALPR on vehicle crops
    import app.services.pipeline as pl

    detection = pl.VehicleDetector().detect(prepared_img)
    if plates := run_fast_alpr_on_crops(detection.vehicles):
        return plates

    # Tier 3 & 4: RapidOCR on vehicle crops, falling back to full-frame RapidOCR
    return _run_rapidocr_fallback(detection, prepared_img, filename)


def recognize_plate_image(
    image_input: str | bytes,
    filename: str = "image.jpg",
) -> RecognitionResponse:
    """Process image through the 4-tier cascaded ANPR pipeline."""
    start_time = time.time()
    resolved_fn = filename or (image_input if isinstance(image_input, str) else "image.jpg")
    prepared_img = decode_image(_resolve_bytes(image_input))
    plates = _execute_cascade(prepared_img, resolved_fn)
    return _build_response(plates, start_time)
```
*(Ensure file is $\le$ 60 lines)*

---

### Step 9: Update Model Warmup in `app/server.py`
In `app/server.py`, warm up both Fast-ALPR and RapidOCR models during server lifespan:

```python
# In app/server.py lifespan:
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logger.info(f"Starting {constants.PROJECT_NAME} v{constants.VERSION}...")
    try:
        from app.services.ocr.fast_alpr_engine import check_fast_alpr_engine
        from app.services.ocr import PlateRecognizer

        VehicleDetector.get_model()
        PlateRecognizer.check_engine()
        check_fast_alpr_engine()
        logger.info("Fast-ALPR and RapidOCR engines pre-warmed successfully.")
    except Exception as exc:
        logger.warning(f"Non-fatal warmup warning: {exc}")

    yield
    logger.info(f"Shutting down {constants.PROJECT_NAME}...")
```

Also update the route description for `/recognize` in `app/server.py` to remove references to occupancy/humans.

---

## 5. Verification & Testing

### 5.1 Verification Checklist
1. **Tier 1 Test (Fast-ALPR Full-Frame)**:
   - Provide an image with a clear plate (e.g. `tests/images/1.jpg`).
   - Fast-ALPR should detect, validate, and return the plate directly in $\approx 30\text{--}40\text{ ms}$.
   - Ensure the response format is thin:
     ```json
     {
       "results": [{"plate": "RJ09GA0165"}],
       "execution_time_ms": 32.1
     }
     ```
2. **Tier 2 Test (Fast-ALPR on Vehicle Crop)**:
   - Provide an image where the plate is too small in the full frame for Fast-ALPR, but YOLO26 detects the vehicle.
   - Fast-ALPR detects and reads the plate from the cropped vehicle image.
   - Returns valid thin response in $\approx 50\text{--}70\text{ ms}$.
3. **Tier 3 Test (Argus RapidOCR on Vehicle Crop)**:
   - Provide a difficult/unusual font plate where Fast-ALPR yields no candidates.
   - Verify log outputs `Fast-ALPR produced no plates. Falling back to RapidOCR...`.
   - RapidOCR runs on vehicle crop, extracts candidates via 2D spatial pairing, validates with MoRTH, and returns thin response in $\approx 120\text{--}180\text{ ms}$.
4. **Tier 4 Test (Argus RapidOCR Full-Frame Fallback)**:
   - Provide an image where vehicle detection fails or crops incorrectly, but RapidOCR full-frame succeeds.
   - Returns thin response in $\approx 220\text{--}280\text{ ms}$.
5. **Empty Case Test**:
   - Provide an image of an empty road or non-vehicle.
   - Response must be:
     ```json
     {
       "results": [],
       "execution_time_ms": 54.2
     }
     ```
6. **NASA JPL Rule 4 Audit**:
   Run in `repos/argus`:
   ```bash
   wc -l app/schemas.py \
         app/services/ocr/fast_alpr_*.py \
         app/services/pipeline/orchestrator.py \
         app/services/pipeline/response.py \
         app/services/detector/detector.py
   ```
   **Every output must be $\le$ 60 lines.**

---

## 6. Summary of Key Files Changed

| File | Action | Purpose |
| :--- | :--- | :--- |
| `pyproject.toml` | Modify | Add `fast-alpr[onnx]>=0.4.0` dependency. |
| `app/schemas.py` | Modify | Remove human fields; slim `PlateResult` and `RecognitionResponse` to thin format. |
| `app/services/ocr/fast_alpr_engine.py` | **Create** | Fast-ALPR singleton and warmup. |
| `app/services/ocr/fast_alpr_padding.py` | **Create** | Margin padding (15% X, 10% Y) to avoid edge truncation. |
| `app/services/ocr/fast_alpr_runner.py` | **Create** | Fast-ALPR execution on full frame (Tier 1) and vehicle crops (Tier 2). |
| `app/services/pipeline/orchestrator.py` | Modify | 4-tier cascaded fallback orchestrator (`full Fast-ALPR` $\rightarrow$ `crop Fast-ALPR` $\rightarrow$ `crop RapidOCR` $\rightarrow$ `full RapidOCR`). |
| `app/services/pipeline/response.py` | Modify | Thin response constructor (`results`, `execution_time_ms`). |
| `app/services/detector/detector.py` | Modify | Remove human occupancy partitioning; keep vehicle detection for Tier 2/3. |
| `app/server.py` | Modify | Warm up both engines during startup; remove human references. |
