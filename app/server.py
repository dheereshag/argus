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
    """Pre-warms YOLO model and verifies OCR engines during service startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    try:
        VehicleDetector.get_model()
        PlateRecognizer.check_engine()
        logger.info("AI models initialized and verified successfully.")
    except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
        logger.warning(f"Non-fatal warning warming models during startup: {exc}")

    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Enterprise Automatic Number Plate Recognition (ANPR) Microservice.",
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
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
        return response

    @app.exception_handler(ANPRServiceError)
    async def handle_anpr_error(_: Request, exc: ANPRServiceError) -> JSONResponse:
        return _error_response(exc.status_code, exc.message, exc.__class__.__name__)

    @app.exception_handler(ContractViolation)
    async def handle_contract_error(_: Request, exc: ContractViolation) -> JSONResponse:
        return _error_response(500, "Internal system assertion contract failed.", "ContractViolation", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_val_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(422, "Request validation error.", "RequestValidationError", exc.errors())

    @app.exception_handler(Exception)
    async def handle_generic_error(req: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {req.url.path}: {exc}")
        return _error_response(500, "An internal server error occurred.", "InternalServerError")

    @app.get("/", summary="Service Information", tags=["Info"])
    async def root() -> dict[str, str]:
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
        image_bytes = await file.read()
        validate_image_upload(image_bytes, content_type=file.content_type)
        return recognize_plate_image(image_bytes, filename=file.filename or "image.jpg")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=True)
