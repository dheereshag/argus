import asyncio
import json
import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.config import settings, Settings
from app.core.exceptions import (
    ANPRServiceError,
    ProviderNotFoundError,
    InvalidImageError,
    anpr_exception_handler
)

def test_settings_default_values():
    assert settings.PROJECT_NAME == "Argus ANPR Microservice"
    assert settings.VERSION == "1.0.0"
    assert settings.DEFAULT_PROVIDER.value in ["paddleocr", "platerecognizer", "nvidia"]
    assert settings.HUMAN_CONF_THRESH == 0.30
    assert settings.VEHICLE_CONF_THRESH == 0.35

def test_anpr_service_error():
    err = ANPRServiceError("Base error message", status_code=500)
    assert err.message == "Base error message"
    assert err.status_code == 500
    assert str(err) == "Base error message"

def test_provider_not_found_error():
    err = ProviderNotFoundError("invalid_provider", ["paddleocr", "nvidia"])
    assert err.status_code == 400
    assert "Unknown provider 'invalid_provider'" in err.message
    assert "paddleocr, nvidia" in err.message

def test_invalid_image_error():
    err = InvalidImageError("Unsupported image format")
    assert err.status_code == 400
    assert err.message == "Unsupported image format"

def test_anpr_exception_handler():
    scope = {"type": "http", "method": "GET", "path": "/test", "headers": []}
    request = Request(scope=scope)
    exc = InvalidImageError("Test image exception")
    
    response = asyncio.run(anpr_exception_handler(request, exc))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    
    data = json.loads(response.body.decode())
    assert data["success"] is False
    assert data["error"] == "InvalidImageError"
    assert data["detail"] == "Test image exception"
