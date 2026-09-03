from unittest.mock import MagicMock, patch

from app.schemas.plate import RecognitionStatusEnum
from app.services.pipeline import recognize_plate_image


@patch("app.services.pipeline.filter_vehicle_and_occupancy")
def test_recognize_rejected_human(mock_yolo, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
        "status_message": "Image rejected: Human presence detected.",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": True,
    }
    response = recognize_plate_image(sample_image_bytes, filename="car_human.jpg")
    assert response.success is False
    assert response.status == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert response.human_detected is True
    assert response.results == []


@patch("app.services.pipeline.filter_vehicle_and_occupancy")
def test_recognize_rejected_no_four_wheeler(mock_yolo, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
        "status_message": "Image rejected: No 4-wheeler vehicle detected.",
        "vehicle_detected": False,
        "vehicle_type": None,
        "human_detected": False,
    }
    response = recognize_plate_image(sample_image_bytes, filename="scenery.jpg")
    assert response.success is False
    assert response.status == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    assert response.vehicle_detected is False


@patch("app.services.pipeline.filter_vehicle_and_occupancy")
def test_recognize_rejected_multiple_vehicles(mock_yolo, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
        "status_message": "Image rejected: Multiple 4-wheeler vehicles detected (2 vehicles).",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": False,
    }
    response = recognize_plate_image(sample_image_bytes, filename="two_cars.jpg")
    assert response.success is False
    assert response.status == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert response.vehicle_detected is True
    assert response.results == []


@patch("app.services.pipeline.DoclingStrategy")
@patch("app.services.pipeline.filter_vehicle_and_occupancy")
def test_recognize_success(mock_yolo, mock_docling_cls, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": True,
        "status": None,
        "status_message": "Eligible vehicle.",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": False,
    }

    mock_docling = MagicMock()
    mock_docling.recognize.return_value = [{"plate": "RJ09GA0165", "state": "Rajasthan"}]
    mock_docling_cls.return_value = mock_docling

    response = recognize_plate_image(sample_image_bytes, filename="car.jpg")
    assert response.success is True
    assert response.status == RecognitionStatusEnum.SUCCESS
    assert len(response.results) == 1
    assert response.results[0].plate == "RJ09GA0165"
    assert response.results[0].state == "Rajasthan"


@patch("app.services.pipeline.DoclingStrategy")
@patch("app.services.pipeline.filter_vehicle_and_occupancy")
def test_recognize_no_vehicle_detected(mock_yolo, mock_docling_cls, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": True,
        "status": None,
        "status_message": "No vehicle detected. Eligible for direct plate recognition.",
        "vehicle_detected": False,
        "vehicle_type": None,
        "human_detected": False,
    }

    mock_docling = MagicMock()
    mock_docling.recognize.return_value = [{"plate": "DL01AB1234", "state": "Delhi"}]
    mock_docling_cls.return_value = mock_docling

    response = recognize_plate_image(sample_image_bytes, filename="plate_crop.jpg")
    assert response.success is True
    assert response.status == RecognitionStatusEnum.SUCCESS
    assert response.vehicle_detected is False
    assert len(response.results) == 1
    assert response.results[0].plate == "DL01AB1234"
    assert "License plate successfully detected and recognized via docling." in response.status_message
