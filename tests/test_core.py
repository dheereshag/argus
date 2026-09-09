import pytest

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


def test_contracts_require_and_ensure():
    from app.core.contracts import ContractViolation, ensure, require

    require(True, "should not raise")
    ensure(True, "should not raise")

    with pytest.raises(ContractViolation, match="precondition failed"):
        require(False, "precondition check failed")

    with pytest.raises(ContractViolation, match="postcondition failed"):
        ensure(False, "postcondition check failed")


def test_contracts_bounded():
    from app.core.contracts import ContractViolation, bounded

    assert bounded(None, 5, "empty") == []
    assert bounded([1, 2], 5, "small") == [1, 2]
    assert bounded([1, 2, 3, 4], 2, "truncated") == [1, 2]

    with pytest.raises(ContractViolation, match="limit must be positive"):
        bounded([1], 0, "invalid limit")
