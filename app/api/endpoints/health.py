from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.plate import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check API Health Status")
def check_health():
    return HealthResponse(status="healthy", version=settings.VERSION)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Check Model Readiness",
    description=(
        "Returns 200 when both the YOLO and PaddleOCR model singletons are fully loaded. "
        "Returns 503 while either is still initialising. "
        "Use this as a Kubernetes readinessProbe target to avoid routing traffic to a pod "
        "that is still warming up."
    ),
)
def check_ready():
    # Import the private singletons directly — they are None until get_*() has been
    # called successfully, which main.py's lifespan does before serving traffic.
    from app.services.strategies.paddle_ocr import _PADDLE_OCR_INSTANCE  # noqa: PLC0415
    from app.services.yolo_filter import _YOLO_MODEL  # noqa: PLC0415

    yolo_ready = _YOLO_MODEL is not None
    paddle_ready = _PADDLE_OCR_INSTANCE is not None
    all_ready = yolo_ready and paddle_ready

    body = ReadyResponse(
        ready=all_ready,
        models={
            "yolo": "loaded" if yolo_ready else "not_loaded",
            "paddleocr": "loaded" if paddle_ready else "not_loaded",
        },
    )

    if not all_ready:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body
