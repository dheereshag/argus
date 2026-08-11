from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ProviderNotFoundError
from app.schemas.plate import ProviderEnum, RecognitionStatusEnum
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.services.factory import PlateRecognizerFactory
from app.services.strategies.nvidia_vision import NvidiaVisionStrategy
from app.services.strategies.paddle_ocr import PaddleOCRStrategy
from app.services.strategies.plate_recognizer import PlateRecognizerStrategy
from app.services.yolo_filter import filter_vehicle_and_occupancy


class DummyStrategy(BasePlateRecognizer):
    """Minimal concrete subclass for factory and parse tests."""

    def _recognize_single_image(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg",
    ) -> List[Dict[str, Any]]:
        return [{"plate": "MH12AB1234", "state": "Maharashtra"}]

    def recognize(
        self,
        image_input: Union[str, bytes],
        filename: str = "image.jpg",
        vehicle_box: Optional[Tuple[int, int, int, int]] = None,
        vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        return [{"plate": "MH12AB1234", "state": "Maharashtra"}]

def test_indian_plate_regex_and_state_codes():
    # Valid Indian plate patterns
    plates_to_test = [
        ("RJ09GA0165", "Rajasthan"),
        ("MH12AB1234", "Maharashtra"),
        ("DL01C1234", "Delhi"),
        ("KA05MB9999", "Karnataka"),
        ("TN07AZ0001", "Tamil Nadu"),
    ]
    for plate_str, expected_state in plates_to_test:
        match = INDIAN_PLATE_REGEX.search(plate_str)
        assert match is not None
        state_code = match.group(1) or match.group(6)
        assert STATE_CODES.get(state_code) == expected_state

def test_base_plate_recognizer_parse_plate_info():
    strategy = DummyStrategy()

    # Test valid regex match
    info = strategy.parse_plate_info("  rj 09 ga 0165 ")
    assert info is not None
    assert info["plate"] == "RJ09GA0165"
    assert info["state"] == "Rajasthan"

    # Test none / empty input
    assert strategy.parse_plate_info("") is None
    assert strategy.parse_plate_info(None) is None  # type: ignore

    # Test invalid plate input returns None (no unvalidated fallback)
    assert strategy.parse_plate_info("XX999999") is None

def test_factory_list_and_get():
    providers = PlateRecognizerFactory.list_providers()
    assert ProviderEnum.PADDLEOCR in providers
    assert ProviderEnum.NVIDIA in providers
    assert ProviderEnum.PLATERECOGNIZER in providers

    # Valid get
    recognizer = PlateRecognizerFactory.get_recognizer(ProviderEnum.PADDLEOCR)
    assert isinstance(recognizer, PaddleOCRStrategy)

    # String input get
    recognizer_str = PlateRecognizerFactory.get_recognizer("nvidia")
    assert isinstance(recognizer_str, NvidiaVisionStrategy)

    # Unknown provider string throws ProviderNotFoundError
    with pytest.raises(ProviderNotFoundError):
        PlateRecognizerFactory.get_recognizer("unknown_provider")

def test_factory_custom_registration():
    PlateRecognizerFactory.register_strategy(ProviderEnum.PADDLEOCR, DummyStrategy)
    rec = PlateRecognizerFactory.get_recognizer(ProviderEnum.PADDLEOCR)
    assert isinstance(rec, DummyStrategy)
    # Restore original strategy
    PlateRecognizerFactory.register_strategy(ProviderEnum.PADDLEOCR, PaddleOCRStrategy)

@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_eligible_vehicle(mock_get_model, sample_image_bytes):
    mock_box_car = MagicMock()
    mock_box_car.__len__.return_value = 1
    mock_box_car.cls.cpu().numpy.return_value = [2]  # Class 2 = car
    mock_box_car.conf.cpu().numpy.return_value = [0.90]
    mock_box_car.xyxy.cpu().numpy.return_value = [[10, 10, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_car

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    res = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res["is_eligible"] is True
    assert res["vehicle_detected"] is True
    assert res["vehicle_type"] == "car"
    assert res["human_detected"] is False

@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_rejected_human(mock_get_model, sample_image_bytes):
    mock_box_human = MagicMock()
    mock_box_human.__len__.return_value = 2
    mock_box_human.cls.cpu().numpy.return_value = [0, 2]  # Class 0 = person, 2 = car
    mock_box_human.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_human.xyxy.cpu().numpy.return_value = [[5, 5, 15, 15], [10, 10, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_human

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    res = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res["is_eligible"] is False
    assert res["status"] == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res["human_detected"] is True

@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_rejected_no_four_wheeler(mock_get_model, sample_image_bytes):
    mock_box_empty = MagicMock()
    mock_box_empty.__len__.return_value = 0
    mock_box_empty.cls.cpu().numpy.return_value = []
    mock_box_empty.conf.cpu().numpy.return_value = []
    mock_box_empty.xyxy.cpu().numpy.return_value = []

    mock_results = MagicMock()
    mock_results.boxes = mock_box_empty

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    res = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res["is_eligible"] is False
    assert res["status"] == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    assert res["vehicle_detected"] is False

@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_rejected_multiple_vehicles(mock_get_model, sample_image_bytes):
    mock_box_multiple = MagicMock()
    mock_box_multiple.__len__.return_value = 2
    mock_box_multiple.cls.cpu().numpy.return_value = [2, 7]  # Class 2 = car, 7 = truck
    mock_box_multiple.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_multiple.xyxy.cpu().numpy.return_value = [[10, 10, 50, 50], [50, 50, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box_multiple

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    res = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res["is_eligible"] is False
    assert res["status"] == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert res["vehicle_detected"] is True
    assert res["vehicle_count"] == 2

@patch("app.services.strategies.paddle_ocr.get_paddle_ocr_engine")
def test_paddle_ocr_strategy_mocked(mock_get_engine, sample_image_bytes):
    mock_engine = MagicMock()
    # Mock return value of engine.ocr: [[ [box, ("RJ09GA0165", 0.95)] ]]
    mock_engine.ocr.return_value = [[
        [None, ("RJ09GA0165", 0.95)]
    ]]
    mock_get_engine.return_value = mock_engine

    strategy = PaddleOCRStrategy(bottom_crop_ratio=0.5)
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"

@patch("requests.post")
def test_nvidia_vision_strategy_mocked(mock_post, sample_image_bytes):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": "The vehicle license plate is RJ09GA0165."}}
        ]
    }
    mock_post.return_value = mock_resp

    strategy = NvidiaVisionStrategy(api_key="test_key")
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"

@patch("requests.post")
def test_plate_recognizer_strategy_mocked(mock_post, sample_image_bytes):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "plate": "rj09ga0165",
                "candidates": [{"plate": "rj09ga0165"}]
            }
        ]
    }
    mock_post.return_value = mock_resp

    strategy = PlateRecognizerStrategy(token="test_token")
    results = strategy.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"


def test_plate_recognizer_skips_large_file():
    recognizer = PlateRecognizerStrategy(token="dummy_token")
    large_bytes = b"0" * int(3.6 * 1024 * 1024)
    res = recognizer._recognize_single_image(large_bytes)
    assert res == []
