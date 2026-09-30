"""Fast-ALPR Model Singleton and Warmup Engine."""

import threading
from typing import Any

from app.core.logging import logger

_alpr_instance: Any = None
_lock = threading.Lock()


def get_fast_alpr_engine() -> Any:
    """Return thread-safe singleton instance of Fast-ALPR engine."""
    global _alpr_instance
    if _alpr_instance is None:
        with _lock:
            if _alpr_instance is None:
                from fast_alpr import ALPR

                logger.info("Initializing Fast-ALPR ONNX models...")
                _alpr_instance = ALPR(
                    detector_model="yolo-v9-s-608-license-plate-end2end",
                    detector_conf_thresh=0.25,
                    ocr_model="cct-xs-v2-global-model",
                )
    return _alpr_instance


def check_fast_alpr_engine() -> bool:
    """Pre-warm and verify Fast-ALPR models during lifespan startup."""
    try:
        engine = get_fast_alpr_engine()
        return engine is not None
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.error(f"Fast-ALPR warmup failed: {exc}")
        return False
