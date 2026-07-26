import pytest
from unittest.mock import MagicMock, patch
from app.schemas.plate import RecognitionStatusEnum, ProviderEnum

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Argus ANPR Microservice"
    assert data["status"] == "online"

def test_list_providers_endpoint(client):
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "available_providers" in data
    assert "default_provider" in data
    assert len(data["available_providers"]) >= 1

def test_recognize_invalid_file_type(client):
    files = {"file": ("test.txt", b"plain text data", "text/plain")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "InvalidImageError"

@patch("app.api.endpoints.recognition.filter_vehicle_and_occupancy")
def test_recognize_rejected_human(mock_yolo, client, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_HUMAN_DETECTED,
        "status_message": "Image rejected: Human presence detected.",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": True
    }
    files = {"file": ("car_human.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "rejected_human_detected"
    assert data["human_detected"] is True
    assert data["results"] == []

@patch("app.api.endpoints.recognition.filter_vehicle_and_occupancy")
def test_recognize_rejected_no_four_wheeler(mock_yolo, client, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER,
        "status_message": "Image rejected: No 4-wheeler vehicle detected.",
        "vehicle_detected": False,
        "vehicle_type": None,
        "human_detected": False
    }
    files = {"file": ("scenery.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "rejected_no_four_wheeler"
    assert data["vehicle_detected"] is False

@patch("app.api.endpoints.recognition.filter_vehicle_and_occupancy")
def test_recognize_rejected_multiple_vehicles(mock_yolo, client, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": False,
        "status": RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES,
        "status_message": "Image rejected: Multiple 4-wheeler vehicles detected (2 vehicles).",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": False
    }
    files = {"file": ("two_cars.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["status"] == "rejected_multiple_vehicles"
    assert data["vehicle_detected"] is True
    assert data["results"] == []

@patch("app.api.endpoints.recognition.PlateRecognizerFactory.get_recognizer")
@patch("app.api.endpoints.recognition.filter_vehicle_and_occupancy")
def test_recognize_success_primary_provider(mock_yolo, mock_get_recognizer, client, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": True,
        "status": None,
        "status_message": "Eligible vehicle.",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": False
    }

    mock_paddle = MagicMock()
    mock_paddle.recognize.return_value = [
        {"plate": "RJ09GA0165", "state": "Rajasthan"}
    ]
    mock_get_recognizer.return_value = mock_paddle

    files = {"file": ("car.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "success"
    assert data["provider"] == "paddleocr"
    assert len(data["results"]) == 1
    assert data["results"][0]["plate"] == "RJ09GA0165"
    assert data["results"][0]["state"] == "Rajasthan"

@patch("app.api.endpoints.recognition.PlateRecognizerFactory.get_recognizer")
@patch("app.api.endpoints.recognition.filter_vehicle_and_occupancy")
def test_recognize_fallback_to_nvidia(mock_yolo, mock_get_recognizer, client, sample_image_bytes):
    mock_yolo.return_value = {
        "is_eligible": True,
        "status": None,
        "status_message": "Eligible vehicle.",
        "vehicle_detected": True,
        "vehicle_type": "car",
        "human_detected": False
    }

    mock_paddle = MagicMock()
    mock_paddle.recognize.return_value = []  # PaddleOCR finds no plate

    mock_nvidia = MagicMock()
    mock_nvidia.recognize.return_value = [
        {"plate": "MH12AB1234", "state": "Maharashtra"}
    ]

    # Return paddle first, then nvidia
    mock_get_recognizer.side_effect = [mock_paddle, mock_nvidia]

    files = {"file": ("car.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/recognize", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "success"
    assert data["provider"] == "nvidia"
    assert len(data["results"]) == 1
    assert data["results"][0]["plate"] == "MH12AB1234"
    assert data["results"][0]["state"] == "Maharashtra"
