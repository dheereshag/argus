"""System tuning parameters, server defaults, and operational thresholds."""

import os
from importlib.metadata import PackageNotFoundError, version

try:
    VERSION = version("argus")
except PackageNotFoundError:
    VERSION = "dev"

PROJECT_NAME = "Argus ANPR Microservice"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
DOCS_URL = "/docs"
REDOC_URL = "/redoc"
CORS_ORIGINS: list[str] = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS: list[str] = ["*"]
CORS_ALLOW_HEADERS: list[str] = ["*"]

# YOLO Detection & Tuning
YOLO_MODEL_NAME = "yolo26n.pt"
YOLO_CONFIG_DIR = ".cache/ultralytics"
HUMAN_CONF_THRESH = 0.30
VEHICLE_CONF_THRESH = 0.35
DEFAULT_YOLO_IMGSZ = 640
YOLO_AGNOSTIC_NMS = True

# Spatial Geometry Filtering
MIN_HUMAN_BOX_AREA_RATIO = 0.005
MIN_VEHICLE_BOX_AREA_RATIO = 0.01
VEHICLE_IOU_THRESH = 0.50

# Concurrency & Engine Settings
MAX_CONCURRENT_INFERENCES = 4
ONNX_NUM_THREADS = 4

# Image Payload & Security Limits
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000

# Ensure Ultralytics cache directory is writable up front
_yolo_dir = os.path.abspath(YOLO_CONFIG_DIR)
os.makedirs(_yolo_dir, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = _yolo_dir
