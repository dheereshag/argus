"""RapidOCR engine initialization and raw bounding token extraction."""

import logging
from typing import Any

import numpy as np
from PIL import Image
from rapidocr import RapidOCR

from app.core.logging import logger
from app.schemas import OCRToken
from app.services.ocr.geometry import get_box_geometry

_rapid_logger = logging.getLogger("RapidOCR")
_rapid_logger.setLevel(logging.ERROR)
for _h in _rapid_logger.handlers:
    _h.setLevel(logging.ERROR)

_engine: RapidOCR | None = None


def get_engine() -> RapidOCR:
    """Return singleton RapidOCR engine instance, creating it on first access."""
    global _engine
    if _engine is None:
        from app.core.config import settings

        params = {"EngineConfig.onnxruntime.intra_op_num_threads": settings.ONNX_NUM_THREADS}
        _engine = RapidOCR(params=params)
    return _engine


def check_engine() -> bool:
    """Verify that the RapidOCR engine is operational during startup health checks."""
    return get_engine() is not None


def run_ocr_inference(img_pil: Image.Image, engine_or_getter: Any = None) -> list[OCRToken]:
    """Execute RapidOCR engine and return bounding tokens with geometry."""
    try:
        eng = engine_or_getter() if callable(engine_or_getter) else (engine_or_getter or get_engine())
        res = eng(np.array(img_pil))
        txts = getattr(res, "txts", None) if res else None
        scores = getattr(res, "scores", None) if res else None
        if not txts or not scores:
            return []
        boxes = getattr(res, "boxes", None)
        tokens: list[OCRToken] = []
        for idx, (t, s) in enumerate(zip(txts, scores, strict=False)):
            raw_b = boxes[idx] if (boxes is not None and idx < len(boxes)) else None
            cx, cy, bbox = get_box_geometry(raw_b)
            tokens.append(OCRToken(text=str(t), score=float(s), cx=cx, cy=cy, box=bbox))
        if any(item.cy is not None for item in tokens):
            tokens.sort(key=lambda x: x.cy if x.cy is not None else 9999.0)
        return tokens
    except (RuntimeError, ValueError, TypeError, IndexError, AttributeError, OSError) as e:
        logger.error(f"[ocr/rapidocr] OCR execution failed: {e}")
        return []
