from datetime import UTC, datetime

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import logger
from app.schemas.health import ComponentHealth, HealthResponse, HealthStatusEnum
from app.services.strategies.docling_ocr import check_docling_engine
from app.services.yolo_filter import get_yolo_model

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Check the operational status of the service, YOLO model, OCR engine, and configured providers.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    components: dict[str, ComponentHealth] = {}
    overall_status = HealthStatusEnum.HEALTHY

    # 1. YOLO v11 Model Check
    try:
        # get_yolo_model() either returns a real model or raises (ensure()'d
        # contract) — no None case to branch on; a load failure lands in the
        # except below.
        get_yolo_model()
        components["yolo"] = ComponentHealth(
            status=HealthStatusEnum.HEALTHY,
            details="YOLO v11 model is loaded and ready.",
            metadata={
                "model_name": settings.YOLO_MODEL_NAME,
                "vehicle_conf_thresh": settings.VEHICLE_CONF_THRESH,
                "human_conf_thresh": settings.HUMAN_CONF_THRESH,
            },
        )
    except (RuntimeError, ValueError, OSError, AttributeError) as exc:
        logger.error(f"Health check failed for YOLO component: {exc}")
        components["yolo"] = ComponentHealth(
            status=HealthStatusEnum.UNHEALTHY,
            details=f"YOLO error: {exc}",
        )
        overall_status = HealthStatusEnum.DEGRADED

    # 2. Docling (RapidOCR) Engine Check
    try:
        docling_ok = check_docling_engine()
        if docling_ok:
            components["docling"] = ComponentHealth(
                status=HealthStatusEnum.HEALTHY,
                details="Docling RapidOCR engine is operational.",
                metadata={"engine": "rapidocr"},
            )
        else:
            components["docling"] = ComponentHealth(
                status=HealthStatusEnum.DEGRADED,
                details="Docling RapidOCR engine failed verification.",
            )
            overall_status = HealthStatusEnum.DEGRADED
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.error(f"Health check failed for Docling component: {exc}")
        components["docling"] = ComponentHealth(
            status=HealthStatusEnum.UNHEALTHY,
            details=f"Docling error: {exc}",
        )
        overall_status = HealthStatusEnum.DEGRADED

    return HealthResponse(
        status=overall_status,
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        timestamp=datetime.now(UTC),
        components=components,
    )
