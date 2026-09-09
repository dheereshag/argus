"""Tests for app/main.py CLI entrypoint."""

import sys
from unittest.mock import patch

import pytest

from app.core.exceptions import ANPRServiceError
from app.main import main
from app.schemas import RecognitionResponse, RecognitionStatusEnum


def _sample_response():
    return RecognitionResponse(
        success=True,
        rejected=False,
        status=RecognitionStatusEnum.SUCCESS,
        status_message="Plate detected",
        vehicle_detected=True,
        vehicle_type="car",
        human_detected=False,
        filename="test.jpg",
        results=[],
    )


@patch("app.main.recognize_plate_image")
@patch("app.main.PlateRecognizer.check_engine")
@patch("app.main.VehicleDetector.get_model")
def test_main_cli_success(mock_get_model, mock_check_engine, mock_recognize, monkeypatch: pytest.MonkeyPatch):
    """Test successful CLI execution with valid image argument."""
    monkeypatch.setattr(sys, "argv", ["app.main", "test.jpg"])
    mock_recognize.return_value = _sample_response()

    main()

    mock_get_model.assert_called_once()
    mock_check_engine.assert_called_once()
    mock_recognize.assert_called_once_with("test.jpg")


@patch("app.main.recognize_plate_image")
@patch("app.main.PlateRecognizer.check_engine")
@patch("app.main.VehicleDetector.get_model")
def test_main_cli_yolo_warning(mock_get_model, mock_check_engine, mock_recognize, monkeypatch: pytest.MonkeyPatch):
    """Test CLI handles non-fatal YOLO warmup warnings gracefully."""
    monkeypatch.setattr(sys, "argv", ["app.main", "test.jpg"])
    mock_get_model.side_effect = RuntimeError("GPU memory warning")
    mock_recognize.return_value = _sample_response()

    main()

    mock_recognize.assert_called_once_with("test.jpg")


@patch("app.main.recognize_plate_image")
@patch("app.main.PlateRecognizer.check_engine")
@patch("app.main.VehicleDetector.get_model")
def test_main_cli_service_error_exits(
    mock_get_model, mock_check_engine, mock_recognize, monkeypatch: pytest.MonkeyPatch
):
    """Test CLI logs error and exits with code 1 when recognition fails."""
    monkeypatch.setattr(sys, "argv", ["app.main", "bad_image.jpg"])
    mock_recognize.side_effect = ANPRServiceError("Processing failed", status_code=500)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
