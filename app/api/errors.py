"""Standardized JSON error responses and exception handler registration."""

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.contracts import ContractViolation
from app.core.exceptions import ANPRServiceError
from app.core.logging import logger
from app.schemas import APIErrorResponse


def create_error_response(
    status_code: int,
    message: str,
    error_type: str,
    details: Any = None,
) -> JSONResponse:
    """Construct a standardized JSON error response adhering to APIErrorResponse schema."""
    payload = APIErrorResponse(
        success=False,
        status_code=status_code,
        message=message,
        error_type=error_type,
        details=details,
        timestamp=datetime.now(UTC),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers mapping errors to standardized APIErrorResponse."""
    @app.exception_handler(ANPRServiceError)
    async def handle_anpr_error(_: Request, exc: ANPRServiceError) -> JSONResponse:
        return create_error_response(exc.status_code, exc.message, exc.__class__.__name__)

    @app.exception_handler(ContractViolation)
    async def handle_contract_error(_: Request, exc: ContractViolation) -> JSONResponse:
        return create_error_response(500, "Internal assertion failed.", "ContractViolation", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_val_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return create_error_response(422, "Request validation error.", "RequestValidationError", exc.errors())

    @app.exception_handler(Exception)
    async def handle_generic_error(req: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {req.url.path}: {exc}")
        return create_error_response(500, "An internal server error occurred.", "InternalServerError")
