"""Runtime environment configuration for Argus ANPR."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime environment configuration loaded from .env or environment variables."""

    ENABLE_FULL_FRAME_OCR: bool = False
    INCLUDE_UNIDENTIFIED_VEHICLES: bool = False
    DEBUG_SAVE_CROPS: bool = False
    DEBUG_CROPS_DIR: str = "debug_crops"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
