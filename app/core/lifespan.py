"""Application lifecycle management and model pre-warming."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core import constants
from app.core.logging import logger
from app.services.detector import VehicleDetector
from app.services.ocr import PlateRecognizer
from app.services.ocr.fast_alpr_engine import check_fast_alpr_engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifecycle: warm up AI models on startup and log shutdown."""
    logger.info(f"Starting {constants.PROJECT_NAME} v{constants.VERSION}...")
    try:
        VehicleDetector.get_model()
        PlateRecognizer.check_engine()
        check_fast_alpr_engine()
        logger.info("AI models initialized and verified successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.warning(f"Non-fatal warning warming models during startup: {exc}")

    yield
    logger.info(f"Shutting down {constants.PROJECT_NAME}...")
