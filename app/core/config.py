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
    DEFAULT_PROVIDER: ProviderEnum = ProviderEnum.PLATERECOGNIZER

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
