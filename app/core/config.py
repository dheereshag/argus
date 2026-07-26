import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from app.schemas.plate import ProviderEnum

class Settings(BaseSettings):
    PROJECT_NAME: str = "Argus ANPR Microservice"
    VERSION: str = "1.0.0"

    # API Keys & Endpoints
    PLATE_RECOGNIZER_TOKEN: str = ""
    NVIDIA_API_KEY: str = ""
    NVIDIA_INVOKE_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    DEFAULT_PROVIDER: ProviderEnum = ProviderEnum.PADDLEOCR

    # YOLO Model Settings
    YOLO_MODEL_NAME: str = "yolo11n.pt"
    YOLO_CONFIG_DIR: str = "/tmp/Ultralytics"
    HUMAN_CONF_THRESH: float = 0.30
    VEHICLE_CONF_THRESH: float = 0.35

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
os.environ["YOLO_CONFIG_DIR"] = settings.YOLO_CONFIG_DIR

