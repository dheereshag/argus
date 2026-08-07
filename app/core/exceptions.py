from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger

class ANPRServiceError(Exception):
    """Base exception for ANPR domain errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class ProviderNotFoundError(ANPRServiceError):
    """Raised when an unknown recognition provider is requested."""
    def __init__(self, provider: str, available_providers: list[str]):
        message = f"Unknown provider '{provider}'. Available providers: {', '.join(available_providers)}"
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)

class InvalidImageError(ANPRServiceError):
    """Raised when the uploaded file is not a valid image."""
    def __init__(self, message: str = "Invalid or unsupported image file"):
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)

class PayloadTooLargeError(ANPRServiceError):
    """Raised when an upload exceeds the byte or pixel budget."""
    def __init__(self, message: str = "Uploaded image exceeds the maximum permitted size"):
        super().__init__(message=message, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

async def contract_violation_handler(request: Request, exc: Exception):
    """
    An internal invariant broke (see app/core/contracts.py).

    Logged at error, never warning: unlike a domain error, nothing the client
    did should be able to cause this, so every occurrence is a defect. The
    response deliberately does not echo the contract message — it names
    internals — but the log does.
    """
    logger.error(f"Contract violation on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "InternalContractViolation",
            "detail": "An internal consistency check failed. The request was not processed.",
        },
    )


async def anpr_exception_handler(request: Request, exc: ANPRServiceError):
    logger.warning(f"ANPR Exception [{exc.__class__.__name__}] URL: {request.url} - Detail: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.__class__.__name__,
            "detail": exc.message
        }
    )
