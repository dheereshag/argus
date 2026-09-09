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
    assert result.vehicle_detected is True
    assert result.vehicle_type == "car"
    assert result.human_detected is False
    assert result.vehicle_count == 1
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
    assert res_default.human_detected is True

    # Explicit policy: reject_on_human is False -> eligible
    monkeypatch.setattr(settings, "REJECT_ON_HUMAN_DETECTED", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.human_detected is True
    assert res_allowed.vehicle_detected is True


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
    assert res_default.vehicle_detected is False
    assert res_default.vehicle_count == 0

    # Explicit policy: reject_on_no_vehicle is False -> eligible for direct plate OCR
    monkeypatch.setattr(settings, "REJECT_ON_NO_VEHICLE", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.vehicle_detected is False
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
    assert res_default.vehicle_detected is True
    assert res_default.vehicle_count == 2

    # Explicit policy: reject_on_multiple_vehicles is False -> eligible with primary vehicle
    monkeypatch.setattr(settings, "REJECT_ON_MULTIPLE_VEHICLES", False)
    res_allowed = VehicleDetector().detect(sample_image_bytes)
    assert res_allowed.is_eligible is True
    assert res_allowed.status is None
    assert res_allowed.vehicle_detected is True
    assert res_allowed.vehicle_count == 2


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
    assert result.vehicle_detected is False
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
    assert result.vehicle_detected is False


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


