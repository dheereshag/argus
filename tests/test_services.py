from unittest.mock import MagicMock, patch

import pytest

from app.constants import INDIAN_PLATE_REGEX, STATE_CODES
from app.core.config import settings
from app.schemas import RecognitionStatusEnum
from app.services.detector import VehicleDetector
from app.services.plate_rules import normalize_candidate_strings


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
        match = INDIAN_PLATE_REGEX.fullmatch(plate_str)
        assert match is not None
        if match.group(1):
            assert STATE_CODES.get(match.group(1)) == expected_state
        elif match.group(5):
            assert STATE_CODES.get("BH") == expected_state

    # Invalid patterns
    assert INDIAN_PLATE_REGEX.fullmatch("INVALID123") is None
    assert INDIAN_PLATE_REGEX.fullmatch("XX99YY9999") is None  # Unknown state code XX


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_detection_flow(mock_get_model, sample_image_bytes):
    mock_box = MagicMock()
    mock_box.__len__.return_value = 1
    mock_box.cls.cpu().numpy.return_value = [2]  # Class 2 = car
    mock_box.conf.cpu().numpy.return_value = [0.85]
    mock_box.xyxy.cpu().numpy.return_value = [[10, 10, 50, 50]]

    mock_results = MagicMock()
    mock_results.boxes = mock_box

    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    result = VehicleDetector().detect(sample_image_bytes)
    assert result.is_eligible is True
    assert result.vehicle_type == "car"
    assert result.vehicle_count == 1
    assert result.human_count == 0
    assert result.vehicle_box == (10, 10, 50, 50)
    assert result.crop is not None


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_human_detection_policy(
    mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch
):
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
    res_default = VehicleDetector().detect(sample_image_bytes)
    assert res_default.is_eligible is False
    assert res_default.status == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res_default.human_count == 1

    # Explicit policy: reject_on_human is False -> eligible
    monkeypatch.setattr(settings, "REJECT_ON_HUMAN_DETECTED", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.human_count == 1
    assert res_allowed.vehicle_count == 1


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_no_vehicle_policy(mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch):
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
    res_default = VehicleDetector().detect(sample_image_bytes)
    assert res_default.is_eligible is False
    assert res_default.status == RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER
    assert res_default.vehicle_count == 0

    # Explicit policy: reject_on_no_vehicle is False -> eligible for direct plate OCR
    monkeypatch.setattr(settings, "REJECT_ON_NO_VEHICLE", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.vehicle_count == 0


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_multiple_vehicles_policy(mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch):
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
    res_default = VehicleDetector().detect(sample_image_bytes)
    assert res_default.is_eligible is False
    assert res_default.status == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert res_default.vehicle_count == 2

    # Explicit policy: reject_on_multiple_vehicles is False -> eligible with primary vehicle
    monkeypatch.setattr(settings, "REJECT_ON_MULTIPLE_VEHICLES", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.vehicle_count == 2


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_human_threshold_policies(
    mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch
):
    # 1 human + 1 car
    mock_box_1h = MagicMock()
    mock_box_1h.__len__.return_value = 2
    mock_box_1h.cls.cpu().numpy.return_value = [0, 2]  # Class 0 = person, 2 = car
    mock_box_1h.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_1h.xyxy.cpu().numpy.return_value = [[5, 5, 15, 15], [10, 10, 90, 90]]

    mock_res_1h = MagicMock()
    mock_res_1h.boxes = mock_box_1h

    mock_model = MagicMock()
    mock_model.return_value = [mock_res_1h]
    mock_get_model.return_value = mock_model

    # MAX_ALLOWED_HUMANS = 1 allows driver
    monkeypatch.setattr(settings, "MAX_ALLOWED_HUMANS", 1)
    res_driver = VehicleDetector().detect(sample_image_bytes)
    assert res_driver.is_eligible is True
    assert res_driver.status is None
    assert res_driver.human_count == 1

    # 2 humans + 1 car with MAX_ALLOWED_HUMANS = 1 -> rejected
    mock_box_2h = MagicMock()
    mock_box_2h.__len__.return_value = 3
    mock_box_2h.cls.cpu().numpy.return_value = [0, 0, 2]  # 2 persons, 1 car
    mock_box_2h.conf.cpu().numpy.return_value = [0.85, 0.85, 0.90]
    mock_box_2h.xyxy.cpu().numpy.return_value = [[5, 5, 15, 15], [6, 6, 16, 16], [10, 10, 90, 90]]
    mock_res_2h = MagicMock()
    mock_res_2h.boxes = mock_box_2h
    mock_model.return_value = [mock_res_2h]

    res_rejected = VehicleDetector().detect(sample_image_bytes)
    assert res_rejected.is_eligible is False
    assert res_rejected.status == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res_rejected.human_count == 2

    # 2 humans + 1 car with MAX_ALLOWED_HUMANS = 2 -> allowed
    monkeypatch.setattr(settings, "MAX_ALLOWED_HUMANS", 2)
    res_2h_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_2h_allowed.is_eligible is True
    assert res_2h_allowed.status is None
    assert res_2h_allowed.human_count == 2


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_filter_vehicle_threshold_policies(
    mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch
):
    # 2 vehicles with MAX_ALLOWED_VEHICLES = 2 -> allowed
    mock_box_2v = MagicMock()
    mock_box_2v.__len__.return_value = 2
    mock_box_2v.cls.cpu().numpy.return_value = [2, 7]  # car, truck
    mock_box_2v.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box_2v.xyxy.cpu().numpy.return_value = [[10, 10, 50, 50], [50, 50, 90, 90]]
    mock_res_2v = MagicMock()
    mock_res_2v.boxes = mock_box_2v

    mock_model = MagicMock()
    mock_model.return_value = [mock_res_2v]
    mock_get_model.return_value = mock_model

    monkeypatch.setattr(settings, "MAX_ALLOWED_VEHICLES", 2)
    res_2v = VehicleDetector().detect(sample_image_bytes)
    assert res_2v.is_eligible is True
    assert res_2v.status is None
    assert res_2v.vehicle_count == 2

    # 3 vehicles with MAX_ALLOWED_VEHICLES = 2 -> rejected
    mock_box_3v = MagicMock()
    mock_box_3v.__len__.return_value = 3
    mock_box_3v.cls.cpu().numpy.return_value = [2, 5, 7]  # car, bus, truck
    mock_box_3v.conf.cpu().numpy.return_value = [0.85, 0.88, 0.90]
    mock_box_3v.xyxy.cpu().numpy.return_value = [[10, 10, 30, 30], [35, 35, 60, 60], [65, 65, 90, 90]]
    mock_res_3v = MagicMock()
    mock_res_3v.boxes = mock_box_3v
    mock_model.return_value = [mock_res_3v]

    res_3v = VehicleDetector().detect(sample_image_bytes)
    assert res_3v.is_eligible is False
    assert res_3v.status == RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES
    assert res_3v.vehicle_count == 3

    # 0 vehicles with MIN_ALLOWED_VEHICLES = 0 -> allowed
    mock_box_0v = MagicMock()
    mock_box_0v.__len__.return_value = 0
    mock_box_0v.cls.cpu().numpy.return_value = []
    mock_box_0v.conf.cpu().numpy.return_value = []
    mock_box_0v.xyxy.cpu().numpy.return_value = []
    mock_res_0v = MagicMock()
    mock_res_0v.boxes = mock_box_0v
    mock_model.return_value = [mock_res_0v]

    monkeypatch.setattr(settings, "MIN_ALLOWED_VEHICLES", 0)
    res_0v = VehicleDetector().detect(sample_image_bytes)
    assert res_0v.is_eligible is True
    assert res_0v.status is None
    assert res_0v.vehicle_count == 0


def test_normalize_candidate_strings():
    # State prefix corrections
    assert "WB12AB1234" in normalize_candidate_strings("W812AB1234")
    assert "RJ14GJ4976" in normalize_candidate_strings("RT14G34976")
    assert "RJ09GA0165" in normalize_candidate_strings("RJ09GA0165")


def test_clamp_box_edge_cases():
    detector = VehicleDetector()
    # Invalid box structures
    assert detector._clamp_box(None, 100, 100) is None
    assert detector._clamp_box([], 100, 100) is None
    assert detector._clamp_box([1, 2], 100, 100) is None
    assert detector._clamp_box(["a", "b", "c", "d"], 100, 100) is None

    # Inverted box (x2 < x1, y2 < y1)
    assert detector._clamp_box((50, 50, 10, 10), 100, 100) == (10, 10, 50, 50)

    # Degenerate box (< 8px)
    assert detector._clamp_box((10, 10, 15, 15), 100, 100) is None

    # Clamped bounds
    assert detector._clamp_box((-10, -10, 200, 200), 100, 100) == (0, 0, 100, 100)


def test_load_rgb_rgba_array():
    import numpy as np

    from app.services.image_processing import load_rgb

    arr_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    img = load_rgb(arr_rgba)
    assert img.size == (20, 20)


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_empty_boxes(mock_get_model, sample_image_bytes):
    mock_results = MagicMock()
    mock_results.boxes = None
    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    result = VehicleDetector().detect(sample_image_bytes)
    assert result.vehicle_count == 0


def test_yolo_get_model_direct():
    model = VehicleDetector.get_model()
    assert model is not None


@patch("app.services.detector.VehicleDetector.get_model")
def test_yolo_unclamped_degenerate_box_skipped(mock_get_model, sample_image_bytes):
    mock_box = MagicMock()
    mock_box.__len__.return_value = 1
    mock_box.cls.cpu().numpy.return_value = [2]  # car
    mock_box.conf.cpu().numpy.return_value = [0.9]
    mock_box.xyxy.cpu().numpy.return_value = [[10, 10, 12, 12]]  # degenerate < 8px width/height

    mock_results = MagicMock()
    mock_results.boxes = mock_box
    mock_model = MagicMock()
    mock_model.return_value = [mock_results]
    mock_get_model.return_value = mock_model

    result = VehicleDetector().detect(sample_image_bytes)
    assert result.vehicle_count == 0


def test_inspect_image_unsupported_format_and_bomb():
    import io
    from unittest.mock import patch

    from PIL import Image

    from app.core.exceptions import InvalidImageError, PayloadTooLargeError
    from app.services.image_processing import probe_image

    # Create GIF in memory
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="GIF")
    with pytest.raises(InvalidImageError, match="Unsupported image format"):
        probe_image(buf.getvalue())

    # DecompressionBombError handling
    with (
        patch("PIL.Image.open", side_effect=Image.DecompressionBombError("Bomb detected")),
        pytest.raises(PayloadTooLargeError, match="Image dimensions exceed permitted budget"),
    ):
        probe_image(b"dummy")



def test_load_rgb_grayscale_and_file_paths(tmp_path, sample_image_bytes):
    from unittest.mock import patch

    import numpy as np
    from PIL import Image

    from app.core.exceptions import InvalidImageError, PayloadTooLargeError
    from app.services.image_processing import load_rgb

    # 1. 2D grayscale array
    gray_arr = np.zeros((30, 30), dtype=np.uint8)
    img_gray = load_rgb(gray_arr)
    assert img_gray.size == (30, 30)
    assert img_gray.mode == "RGB"

    # 2. File path string
    file_path = tmp_path / "test_img.jpg"
    file_path.write_bytes(sample_image_bytes)
    img_file = load_rgb(str(file_path))
    assert img_file.size == (100, 100)

    # 3. Nonexistent file path string
    with pytest.raises(InvalidImageError, match="Could not decode uploaded image"):
        load_rgb(str(tmp_path / "nonexistent.jpg"))

    # 4. DecompressionBombError via file path
    with (
        patch("PIL.Image.open", side_effect=Image.DecompressionBombError("Bomb")),
        pytest.raises(PayloadTooLargeError, match="Image dimensions exceed permitted budget"),
    ):
        load_rgb(str(file_path))


