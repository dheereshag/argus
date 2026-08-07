import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from app.schemas.plate import ProviderEnum

class Settings(BaseSettings):
    PROJECT_NAME: str = "Argus ANPR Microservice"
    VERSION: str = "1.0.0"

    # API Keys & Endpoints
    PLATE_RECOGNIZER_TOKEN: str = ""
    LLAMA_API_KEY: str = ""
    NEMOTRON_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    NVIDIA_INVOKE_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    DEFAULT_PROVIDER: ProviderEnum = ProviderEnum.PADDLEOCR

    # YOLO Model Settings
    YOLO_MODEL_NAME: str = "yolo11n.pt"
    YOLO_CONFIG_DIR: str = "/tmp/Ultralytics"
    HUMAN_CONF_THRESH: float = 0.30
    VEHICLE_CONF_THRESH: float = 0.35

    # PaddleOCR Tuning Settings
    PADDLE_CPU_THREADS: int = 4
    PADDLE_USE_ANGLE_CLS: bool = True

    # Outbound HTTP timeouts (seconds). Never leave these unset: a provider that
    # accepts the connection and then stalls will otherwise hang the worker
    # thread indefinitely.
    HTTP_CONNECT_TIMEOUT: float = 3.0
    HTTP_READ_TIMEOUT: float = 10.0

    # Upload limits
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024        # reject the request body above this
    MAX_IMAGE_PIXELS: int = 50_000_000             # decompression-bomb guard (w * h)
    MAX_IMAGE_EDGE_PX: int = 1920                  # downscale longest edge before inference

    # Work bounds (NASA rule 2 — every loop has a fixed upper bound).
    #
    # The recognition waterfall costs, per vehicle box:
    #   5 ROI tiers x 2 (plain + perspective-warped) x N providers
    # so an unbounded box list multiplies the whole pipeline. YOLO will happily
    # return a dozen boxes for a yard with parked vehicles in frame. At a
    # weighbridge only the vehicle on the platform can matter, and boxes are
    # area-sorted, so the largest few are the only plausible candidates.
    MAX_VEHICLE_BOXES: int = 3

    # OCR line pairing is O(n) over detected text lines with a 2-wide and
    # 3-wide window. A busy frame (signage, hoardings, a tarpaulin of text)
    # produces many lines, none of them plates.
    MAX_OCR_LINES: int = 40

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
os.environ["YOLO_CONFIG_DIR"] = settings.YOLO_CONFIG_DIR

