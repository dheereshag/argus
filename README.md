# Argus — ANPR microservice

Automatic number plate recognition for a fixed-camera weighbridge, built with FastAPI, YOLO v11,
and a Strategy/Factory provider layer.

A YOLO v11 pre-screening pass verifies a single 4-wheeler is present and no person is in frame,
then routes the image to an OCR/vision provider (PaddleOCR, NVIDIA Llama-3.2-11B-Vision, or Plate
Recognizer) through a fallback waterfall.

> **Status: pilot, not production.**
> This service has **no authentication**, and its accuracy has **not been measured against ground
> truth** — see [Known limitations](#known-limitations) before relying on any number it produces.
> The plan to get from here to a defensible pilot is in [`PILOT_PLAN.md`](PILOT_PLAN.md).

---

## Contents

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [How recognition works](#how-recognition-works)
- [Evaluating accuracy](#evaluating-accuracy)
- [Code quality gates](#code-quality-gates)
- [Known limitations](#known-limitations)
- [Project docs](#project-docs)

---

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run fastapi dev
```

Interactive API docs: <http://localhost:8000/docs>

### Tests

```bash
uv run pytest
```

### Deployment (Render)

```bash
# Build
uv sync

# Start — Render injects $PORT
uv run uvicorn main:app --host 0.0.0.0 --port $PORT
```

There is no Dockerfile yet. The service runs a single uvicorn worker with default settings.

---

## Configuration

All settings are environment variables, read via `pydantic-settings`. Every one has a default;
only the provider credentials genuinely need setting.

### Credentials

| Variable | Default | Notes |
|---|---|---|
| `PLATE_RECOGNIZER_TOKEN` | `""` | Plate Recognizer cloud API. Billed per call. |
| `NVIDIA_API_KEY` | `""` | NVIDIA vision API. Billed per token + image. |
| `LLAMA_API_KEY` / `NEMOTRON_API_KEY` | `""` | Alternate NVIDIA keys, tried in order. |
| `NVIDIA_INVOKE_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` | |

Without credentials the service still runs — PaddleOCR is local, and the cloud providers are
skipped with a logged error.

### Detection

| Variable | Default | Notes |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolo11n.pt` | |
| `YOLO_CONFIG_DIR` | `/tmp/Ultralytics` | |
| `HUMAN_CONF_THRESH` | `0.30` | Any person above this rejects the frame. **See limitations.** |
| `VEHICLE_CONF_THRESH` | `0.35` | |
| `PADDLE_CPU_THREADS` | `4` | |
| `PADDLE_USE_ANGLE_CLS` | `true` | |

### Limits and timeouts

| Variable | Default | Purpose |
|---|---|---|
| `HTTP_CONNECT_TIMEOUT` | `3.0` s | Outbound provider connect timeout |
| `HTTP_READ_TIMEOUT` | `10.0` s | Outbound provider read timeout |
| `MAX_UPLOAD_BYTES` | `8388608` (8 MB) | Request body cap; over this returns 413 |
| `MAX_IMAGE_PIXELS` | `50000000` | Decompression-bomb guard |
| `MAX_IMAGE_EDGE_PX` | `1920` | Longest edge after downscale |
| `MAX_VEHICLE_BOXES` | `3` | Vehicle crops the waterfall will process |
| `MAX_OCR_LINES` | `40` | OCR text lines considered per crop |
| `ALLOWED_ORIGINS` | `["*"]` | CORS. **Change before exposing this anywhere.** |

The three `MAX_*` work bounds exist because recognition cost multiplies:
`boxes × 5 ROI tiers × 2 warps × providers`. See [`docs/NASA_RULES.md`](docs/NASA_RULES.md).

---

## API

### `GET /health`

```json
{ "status": "healthy", "version": "1.0.0" }
```

> This endpoint currently returns `healthy` unconditionally, including when the models failed to
> load. Do not wire an orchestrator restart policy to it yet.

### `GET /providers`

```json
{
  "available_providers": ["platerecognizer", "nvidia", "paddleocr"],
  "default_provider": "paddleocr"
}
```

### `POST /recognize`

**Form data:** `file` — a JPEG or PNG image.

There are **no query parameters.** Provider selection is not currently exposed; every request
runs the full fallback waterfall in a fixed order. (Earlier versions of this README documented a
`?provider=` parameter that was never implemented — it is tracked as an open issue.)

#### Success

```json
{
  "success": true,
  "status": "success",
  "status_message": "License plate successfully detected and recognized on truck via paddleocr.",
  "vehicle_detected": true,
  "vehicle_type": "truck",
  "human_detected": false,
  "filename": "car.jpg",
  "provider": "paddleocr",
  "results": [{ "plate": "RJ14GT4976", "state": "Rajasthan", "raw_text": "RJ14GT4976" }],
  "execution_time_ms": 1450.23
}
```

#### Rejected before OCR

```json
{
  "success": false,
  "status": "rejected_human_detected",
  "status_message": "Image rejected: Human presence detected.",
  "vehicle_detected": true,
  "vehicle_type": "truck",
  "human_detected": true,
  "filename": "car2.jpg",
  "provider": "paddleocr",
  "results": [],
  "execution_time_ms": 85.12
}
```

#### Status values

| Status | Meaning |
|---|---|
| `success` | Single 4-wheeler, no person, plate recognised |
| `rejected_no_four_wheeler` | No car, bus or truck detected |
| `rejected_human_detected` | A person was detected anywhere in frame |
| `rejected_multiple_vehicles` | More than one 4-wheeler detected |
| `no_plate_detected` | Passed pre-screening, but no plate extracted |

#### Error responses

| Code | When |
|---|---|
| `400` | Not an image, empty file, or undecodable bytes |
| `413` | Body over `MAX_UPLOAD_BYTES`, or pixel count over `MAX_IMAGE_PIXELS` |
| `500` | Internal contract violation — a defect, not a client error |

---

## How recognition works

```
POST /recognize
  │
  ├─ decode_and_downscale        bomb guard, EXIF, downscale to MAX_IMAGE_EDGE_PX
  │
  ├─ filter_vehicle_and_occupancy    YOLO v11 pre-screen
  │    reject if: person present │ 0 vehicles │ >1 vehicle
  │    return: area-sorted, frame-clamped vehicle boxes
  │
  └─ waterfall: paddleocr → nvidia → platerecognizer
       │
       └─ per provider, per box (capped at MAX_VEHICLE_BOXES):
            tier 1  bottom 1/3  → OCR → if no hit, perspective-warp → OCR
            tier 2  bottom 1/2  → OCR → if no hit, perspective-warp → OCR
            tier 3  bottom 2/3  → OCR → if no hit, perspective-warp → OCR
            tier 4  full box    → OCR → if no hit, perspective-warp → OCR
          tier 5    full frame  → OCR
          first regex-validated hit wins
```

The service does **not** locate the plate. It guesses that the plate is somewhere in the lower
portion of the vehicle box and searches a ladder of crops. That guess is the single largest
source of wrong reads, and replacing it with a real plate detector is the highest-leverage change
available — see `PILOT_PLAN.md`.

Candidate text must match the Indian plate pattern in its **entirety** (`fullmatch`, not
`search`). Substring matching previously accepted `GOODYEAR2024` as plate `ODYEAR2024`.

---

## Evaluating accuracy

Accuracy is measured against hand-written ground truth in `data/labels.csv`.

```bash
# 1. Generate a label template from your images
uv run python scripts/make_labels_template.py data/images/train -o data/labels.csv

# 2. Fill in true_plate by hand — see LABELLING.md
#    Leave it EMPTY for images with no legible plate. Those rows are the
#    false-positive test set and they matter most.

# 3. Run the eval
uv run python eval.py --waterfall

# 4. Once you trust the result, save it as the regression baseline
uv run python eval.py --waterfall --write-baseline
```

Reported: exact match rate, wrong-read rate, miss rate, **false-positive rate**, mean character
error rate, precision, and latency percentiles.

> The status counts printed by the older summary table are an **extraction rate** — how often the
> pipeline emitted something plate-shaped. That is not accuracy and must not be quoted as such.

Read [`LABELLING.md`](LABELLING.md) before labelling anything.

---

## Code quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app/
uv run pytest
```

CI runs all four and **fails** on any finding — `pytest` is configured with
`filterwarnings = ["error"]`, so an unexpected `DeprecationWarning` breaks the build rather than
scrolling past.

The recognition path is written against the applicable subset of the
[NASA Power of 10 rules](https://www.perforce.com/blog/kw/NASA-rules-for-developing-safety-critical-code).
[`docs/NASA_RULES.md`](docs/NASA_RULES.md) records which rules were applied, which were
translated for Python, and which were deliberately skipped and why.

Notably, safety checks use `require()` / `ensure()` from `app/core/contracts.py` rather than
`assert`, because `python -O` strips assert statements.

---

## Known limitations

Please read this section before quoting any capability of this service.

**No measured accuracy.** Until `data/labels.csv` exists and `eval.py` has been run against it,
there is no accuracy figure. The `79/83 SUCCESS` in older eval reports is an extraction rate, and
at least one of those "successes" was an Alamy stock-photo watermark read as a plate.

**No authentication or rate limiting.** Anyone who can reach `/recognize` can spend your Plate
Recognizer and NVIDIA credits. Do not expose this to an untrusted network.

**`/health` is not a real health check.** It returns `healthy` even when the models failed to
load.

**The human-detection gate may reject legitimate weighings.** It fires on a person detected
*anywhere* in frame, with no spatial constraint. If your drivers stay in the cab they are visible
through the windscreen. `scripts/check_human_gate.py` measures this on real captures.

**Plate location is guessed, not detected.** See [How recognition works](#how-recognition-works).

**Model weights are an unverified pickle.** `yolo11n.pt` is loaded via `torch.load`, which
executes arbitrary code. It is committed to the repo rather than fetched at runtime, which
contains the risk, but there is no checksum verification.

**Licence plates are personal data.** Plate plus timestamp plus location falls under India's DPDP
Act 2023 and GDPR where an EU nexus exists. There is no retention policy, and images are sent to
third-party providers without a DPA.

---

## Project docs

| Document | What it covers |
|---|---|
| [`PILOT_PLAN.md`](PILOT_PLAN.md) | The route from here to a defensible pilot, as sequenced issues |
| [`LABELLING.md`](LABELLING.md) | Ground-truth format and conventions |
| [`docs/NASA_RULES.md`](docs/NASA_RULES.md) | Power of 10 rule-by-rule verdicts |
| [`ISSUE_AUDIT_SUMMARY.md`](ISSUE_AUDIT_SUMMARY.md) | Production-readiness audit, condensed |
| [`AUDIT-EXPLAINED.md`](AUDIT-EXPLAINED.md) | The same audit, in full and in plain language |
| `docs/issues/` | Ready-to-file issue and PR bodies |
