"""
Application configuration and environment settings for Argus ANPR.

Loads settings from environment variables and `.env` files using Pydantic Settings.
Configures YOLO model paths, detection thresholds, image payload limits, and server parameters.
"""

import os
from importlib.metadata import PackageNotFoundError, version

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the version from the installed package metadata so it stays in sync
# with pyproject.toml automatically. Falls back to "dev" when running from an
# editable install that hasn't been built (e.g. `uv run` without a prior build).
try:
    _PKG_VERSION = version("argus")
except PackageNotFoundError:
    _PKG_VERSION = "dev"


class Settings(BaseSettings):
    """Global configuration settings for the Argus service."""

    PROJECT_NAME: str = "Argus ANPR Microservice"
    VERSION: str = _PKG_VERSION

    # YOLO Model Settings
    YOLO_MODEL_NAME: str = "yolo26n.pt"  # Pre-trained YOLO26 weights file or model identifier
    YOLO_CONFIG_DIR: str = ".cache/ultralytics"  # Local storage directory for Ultralytics cache
    HUMAN_CONF_THRESH: float = 0.30  # Minimum confidence to flag a person presence
    VEHICLE_CONF_THRESH: float = 0.35  # Minimum confidence to recognize a vehicle (car, bus, truck, motorcycle, bicycle)

    # Image payload & upload security limits
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024  # Reject incoming request body larger than 8 MB
    MAX_IMAGE_PIXELS: int = 50_000_000  # Guard against decompression bomb attacks (w * h)

    # Human Cabin Occupancy & Spatial Filtering Settings
    MIN_HUMAN_BOX_AREA_RATIO: float = 0.005  # Ignore background pedestrians smaller than 0.5% frame area
    MIN_VEHICLE_BOX_AREA_RATIO: float = 0.01  # Ignore distant background vehicles smaller than 1.0% frame area
    VEHICLE_IOU_THRESH: float = 0.50  # Suppress duplicate vehicle bounding boxes with IoU above this
    DEFAULT_YOLO_IMGSZ: int = 640  # Inference resolution for YOLO detector
    YOLO_AGNOSTIC_NMS: bool = True  # Suppress cross-class overlapping bounding boxes in YOLO

    # Concurrency & Engine Settings
    MAX_CONCURRENT_INFERENCES: int = 4  # Maximum concurrent requests processed in threadpool
    ONNX_NUM_THREADS: int = 4  # Intra-op thread count for ONNX Runtime (tuned for Cortex-A76 quad-core)

    # OCR Pipeline Settings
    ENABLE_FULL_FRAME_OCR: bool = False  # If False, full-frame OCR only runs if no vehicles were detected

    # Server & CORS Settings
    SERVER_HOST: str = "127.0.0.1"  # Loopback interface (isolates to localhost/co-located services)
    SERVER_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    DOCS_URL: str | None = "/docs"
    REDOC_URL: str | None = "/redoc"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

# Ultralytics resolves YOLO_CONFIG_DIR relative to the current working
# directory and appends its own "Ultralytics" subfolder. A relative path like
# ".cache/ultralytics" is not writable when the CWD differs (e.g. running from
# a service manager or a different shell), which makes ultralytics silently
# fall back to /tmp/Ultralytics. Resolve to an absolute path and create the
# directory up front so the configured location is always writable.
_yolo_config_dir = os.path.abspath(settings.YOLO_CONFIG_DIR)
os.makedirs(_yolo_config_dir, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = _yolo_config_dir
