from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.schemas import RecognitionStatusEnum
from app.services.ocr import PlateRecognizer
from app.services.plate_rules import (
    INDIAN_PLATE_REGEX,
    STATE_CODES,
    normalize_candidate_strings,
)
from app.services.yolo_filter import filter_vehicle_and_occupancy


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


def test_plate_recognizer_parse_plate_info():
    recognizer = PlateRecognizer()

    # Test valid regex match
    info = recognizer.parse_plate_info("  rj 09 ga 0165 ")
    assert info is not None
    assert info["plate"] == "RJ09GA0165"
    assert info["state"] == "Rajasthan"

    # Test none / empty input
    assert recognizer.parse_plate_info("") is None
    assert recognizer.parse_plate_info(None) is None

    # Test invalid plate input returns None (no unvalidated fallback)
    assert recognizer.parse_plate_info("XX999999") is None


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
    assert res["vehicle_box"] == (10, 10, 90, 90)


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_human_detection_policy(mock_get_model, sample_image_bytes):
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

    # Default policy: reject_on_human is True -> rejected
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is False
    assert res_default["status"] == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res_default["human_detected"] is True

    # Explicit policy: reject_on_human is False -> eligible
    res_allowed = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_human=False)
    assert res_allowed["is_eligible"] is True
    assert res_allowed["status"] is None
    assert res_allowed["human_detected"] is True
    assert res_allowed["vehicle_detected"] is True


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_no_vehicle_policy(mock_get_model, sample_image_bytes):
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

    # Default policy: reject_on_no_vehicle is True -> rejected
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is False
    assert res_default["status"] == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    assert res_default["vehicle_detected"] is False
    assert res_default["vehicle_count"] == 0

    # Explicit policy: reject_on_no_vehicle is False -> eligible for direct plate OCR
    res_allowed = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_no_vehicle=False)
    assert res_allowed["is_eligible"] is True
    assert res_allowed["status"] is None
    assert res_allowed["vehicle_detected"] is False
    assert res_allowed["vehicle_count"] == 0


@patch("app.services.yolo_filter.get_yolo_model")
def test_yolo_filter_multiple_vehicles_policy(mock_get_model, sample_image_bytes):
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

    # Default policy: reject_on_multiple_vehicles is True -> rejected
    res_default = filter_vehicle_and_occupancy(sample_image_bytes)
    assert res_default["is_eligible"] is False
    assert res_default["status"] == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert res_default["vehicle_detected"] is True
    assert res_default["vehicle_count"] == 2

    # Explicit policy: reject_on_multiple_vehicles is False -> eligible with primary vehicle
    res_allowed = filter_vehicle_and_occupancy(sample_image_bytes, reject_on_multiple_vehicles=False)
    assert res_allowed["is_eligible"] is True
    assert res_allowed["status"] is None
    assert res_allowed["vehicle_detected"] is True
    assert res_allowed["vehicle_count"] == 2


@patch("app.services.ocr.get_ocr_engine")
def test_plate_recognizer_mocked(mock_get_engine, sample_image_bytes):
    mock_engine = MagicMock()
    mock_engine.return_value = SimpleNamespace(
        txts=["RJ09GA0165"],
        scores=[0.98],
        boxes=[[0, 0, 100, 30]],
    )
    mock_get_engine.return_value = mock_engine

    recognizer = PlateRecognizer()
    results = recognizer.recognize(sample_image_bytes)
    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"


def test_normalize_candidate_strings():
    # State prefix corrections
    assert "WB12AB1234" in normalize_candidate_strings("W812AB1234")
    assert "RJ14GJ4976" in normalize_candidate_strings("RT14G34976")
    assert "RJ09GA0165" in normalize_candidate_strings("RJ09GA0165")

    # Positional character confusions (O/0, I/1, G3/GJ)
    variants = normalize_candidate_strings("RT14G34976")
    assert "RJ14GJ4976" in variants
