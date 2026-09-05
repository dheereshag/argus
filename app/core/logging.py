"""
Centralized logging interface for Argus ANPR.

Re-exports loguru logger for structured, thread-safe application logging.
"""

from loguru import logger

__all__ = ["logger"]
