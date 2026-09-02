from app.core.config import settings
from app.core.exceptions import (
    ANPRServiceError,
    InvalidImageError,
)


def test_settings_default_values():
    assert settings.PROJECT_NAME == "Argus ANPR Microservice"
    assert settings.VERSION  # non-empty — exact value varies with installed package
    assert settings.HUMAN_CONF_THRESH == 0.30
    assert settings.VEHICLE_CONF_THRESH == 0.35


def test_anpr_service_error():
    err = ANPRServiceError("Base error message", status_code=500)
    assert err.message == "Base error message"
    assert err.status_code == 500
    assert str(err) == "Base error message"



def test_invalid_image_error():
    err = InvalidImageError("Unsupported image format")
    assert err.status_code == 400
    assert err.message == "Unsupported image format"
