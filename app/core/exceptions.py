from fastapi import Request, status
from fastapi.responses import JSONResponse

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

async def anpr_exception_handler(request: Request, exc: ANPRServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.__class__.__name__,
            "detail": exc.message
        }
    )
