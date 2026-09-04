import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas.plate import APIErrorResponse
from app.services.ocr import check_ocr_engine
from app.services.yolo_filter import get_yolo_model


def _error_response(status_code: int, message: str, error_type: str, details: Any = None) -> JSONResponse:
    payload = APIErrorResponse(
        success=False,
        status_code=status_code,
        message=message,
        error_type=error_type,
        details=details,
        timestamp=datetime.now(UTC),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-warms YOLO model and verifies OCR engines during service startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")

    try:
        get_yolo_model()
        logger.info("YOLO v11 model loaded and warmed successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError) as exc:
        logger.warning(f"Non-fatal warning warming YOLO model during startup: {exc}")

    try:
        check_ocr_engine()
        logger.info("RapidOCR engine verified successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.warning(f"Non-fatal warning checking OCR engine during startup: {exc}")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Enterprise Automatic Number Plate Recognition (ANPR) Microservice powered by YOLO v11 and RapidOCR.",
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    @app.exception_handler(ANPRServiceError)
    async def handle_anpr_service_error(request: Request, exc: ANPRServiceError) -> JSONResponse:
        logger.warning(f"Domain exception on {request.url.path}: {exc.message} ({exc.status_code})")
        return _error_response(exc.status_code, exc.message, exc.__class__.__name__)

    @app.exception_handler(ContractViolation)
    async def handle_contract_violation(request: Request, exc: ContractViolation) -> JSONResponse:
        logger.error(f"Internal contract violation on {request.url.path}: {exc}")
        return _error_response(500, "Internal system assertion contract failed.", "ContractViolation", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
        return _error_response(422, "Request validation error.", "RequestValidationError", exc.errors())

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
        return _error_response(500, "An internal server error occurred.", "InternalServerError")

    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
    )
