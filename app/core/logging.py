import sys
from pathlib import Path

from loguru import logger

# Directory for log files
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "argus.log"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

def setup_logging(level: str = "INFO"):
    """
    Configures Loguru handlers for stdout and file logging.
    """
    logger.remove()  # Remove standard handler

    # Console stdout handler
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Rotating file handler
    logger.add(
        LOG_FILE,
        format=LOG_FORMAT,
        level=level,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Loguru logger initialized. Output path: {LOG_FILE.resolve()}")

# Pre-setup default logging configuration
setup_logging()

__all__ = ["logger", "setup_logging"]
