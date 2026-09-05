"""
FastAPI HTTP REST Microservice for Argus ANPR.

Exposes REST endpoints for:
  - Service metadata and health status (GET /).
  - Vehicle license plate recognition from uploaded images (POST /recognize).
  - Standardized JSON error envelopes and request execution timing headers.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas import APIErrorResponse, RecognitionResponse
from app.services.detector import VehicleDetector
from app.services.image_processing import validate_image_upload
from app.services.ocr import PlateRecognizer
from app.services.pipeline import recognize_plate_image


def _error_response(status_code: int, message: str, error_type: str, details: Any = None) -> JSONResponse:
    """
    Construct a standardized JSON error response adhering to APIErrorResponse schema.

    Args:
        status_code: HTTP status code to return.
        message: Human-readable error description.
        error_type: Classification string or exception class name.
        details: Optional contextual or validation error payload.

    Returns:
        JSONResponse: FastAPI response with serialized error payload.
    """
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
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Manage application lifecycle: warm up AI models on startup and log shutdown.

    Pre-loading model weights during startup ensures the initial inference request
    does not incur cold-start latency spikes.
    """
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    try:
        # Pre-warm YOLO v11 model weights and verify RapidOCR engine
        VehicleDetector.get_model()
        PlateRecognizer.check_engine()
        logger.info("AI models initialized and verified successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.warning(f"Non-fatal warning warming models during startup: {exc}")

    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Configures:
      - Lifespan management for model pre-warming.
      - CORS middleware.
      - Execution timing middleware (attaching 'X-Process-Time-Ms' header).
      - Domain exception handlers mapping errors to APIErrorResponse.
      - Service endpoints: GET / and POST /recognize.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Enterprise Automatic Number Plate Recognition (ANPR) Microservice.",
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS) middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # --------------------------------------------------------------------------
    # Request Timing Middleware
    # --------------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Measures total HTTP request roundtrip time and sets X-Process-Time-Ms header."""
        start_time = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
        return response

    # --------------------------------------------------------------------------
    # Exception Handlers
    # --------------------------------------------------------------------------
    @app.exception_handler(ANPRServiceError)
    async def handle_anpr_error(_: Request, exc: ANPRServiceError) -> JSONResponse:
        """Handle domain-specific ANPR service errors (e.g. 400 Bad Request, 413 Payload Too Large)."""
        return _error_response(exc.status_code, exc.message, exc.__class__.__name__)

    @app.exception_handler(ContractViolation)
    async def handle_contract_error(_: Request, exc: ContractViolation) -> JSONResponse:
        """Handle runtime precondition or postcondition contract violations as 500 Internal Server Errors."""
        return _error_response(500, "Internal system assertion contract failed.", "ContractViolation", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_val_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle FastAPI / Pydantic request parameter validation failures (422 Unprocessable Entity)."""
        return _error_response(422, "Request validation error.", "RequestValidationError", exc.errors())

    @app.exception_handler(Exception)
    async def handle_generic_error(req: Request, exc: Exception) -> JSONResponse:
        """Catch-all unhandled exception handler logging error trace and returning 500."""
        logger.exception(f"Unhandled server error on {req.url.path}: {exc}")
        return _error_response(500, "An internal server error occurred.", "InternalServerError")

    # --------------------------------------------------------------------------
    # API Endpoints
    # --------------------------------------------------------------------------
    @app.get("/", summary="Service Information", tags=["Info"])
    async def root() -> dict[str, str]:
        """Return microservice name, version, status, and link to interactive documentation."""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "running",
            "docs": "/docs",
        }

    @app.post("/recognize", response_model=RecognitionResponse, summary="Recognize Vehicle License Plate", tags=["Recognition"])
    async def recognize_plate(
        file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, BMP)")],
    ) -> RecognitionResponse:
        """
        Process an uploaded vehicle image through the Two-Stage ANPR Pipeline.

        Validates MIME type and dimensions, runs YOLO vehicle detection and weighbridge
        occupancy gatekeeping, performs RapidOCR plate recognition, and returns
        extracted plate numbers and state registrations.
        """
        image_bytes = await file.read()
        validate_image_upload(image_bytes, content_type=file.content_type)
        return recognize_plate_image(image_bytes, filename=file.filename or "image.jpg")

    return app


# Default application instance for ASGI servers (e.g. uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=True)
