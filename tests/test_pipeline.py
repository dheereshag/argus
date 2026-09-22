from unittest.mock import MagicMock, patch

from PIL import Image

from app.schemas import DetectedVehicle, DetectionResult
from app.services.pipeline import recognize_plate_image


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_human_and_vehicle(mock_yolo, mock_ocr_cls, sample_image_bytes):
    mock_yolo.return_value = DetectionResult(
        vehicles=[DetectedVehicle(vehicle_type="car", box=(10, 10, 90, 90))],
        humans_outside=1,
        humans_inside=0,
    )
    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = [{"plate": "RJ09GA0165", "state": "Rajasthan"}]
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="car_human.jpg")
    assert response.humans_outside == 1
    assert response.humans_inside == 0
    assert len(response.results) == 1
    assert response.results[0].plate == "RJ09GA0165"
    assert response.results[0].vehicle_type == "car"


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_no_vehicle_no_plate(mock_yolo, mock_ocr_cls, sample_image_bytes):
    mock_yolo.return_value = DetectionResult(
        vehicles=[],
        humans_outside=0,
        humans_inside=0,
    )
    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = []
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="scenery.jpg")
    assert response.results == []
    assert response.humans_outside == 0


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_multiple_vehicles_success(mock_yolo, mock_ocr_cls, sample_image_bytes):
    crop_car = Image.new("RGB", (60, 60))
    crop_truck = Image.new("RGB", (80, 80))
    mock_yolo.return_value = DetectionResult(
        vehicles=[
            DetectedVehicle(
                vehicle_type="car",
                box=(10, 10, 70, 70),
                crop=crop_car,
                crop_box=(10, 10, 70, 70),
            ),
            DetectedVehicle(
                vehicle_type="truck",
                box=(100, 100, 180, 180),
                crop=crop_truck,
                crop_box=(100, 100, 180, 180),
            ),
        ],
        humans_outside=0,
        humans_inside=1,
    )

    mock_ocr = MagicMock()
    mock_ocr.recognize.side_effect = [
        [{"plate": "DL01AB1234", "state": "Delhi", "box": (5, 5, 25, 15)}],
        [{"plate": "MH12CD5678", "state": "Maharashtra", "box": (10, 10, 40, 25)}],
    ]
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="multi_vehicles.jpg")
    assert response.humans_outside == 0
    assert response.humans_inside == 1
    assert len(response.results) == 2
    assert response.results[0].plate == "DL01AB1234"
    assert response.results[0].vehicle_type == "car"
    assert response.results[0].box == (15, 15, 35, 25)
    assert response.results[1].plate == "MH12CD5678"
    assert response.results[1].vehicle_type == "truck"
    assert response.results[1].box == (110, 110, 140, 125)


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_vehicle_cropped(mock_yolo, mock_ocr_cls, sample_image_bytes):
    dummy_crop = Image.new("RGB", (70, 70))
    mock_yolo.return_value = DetectionResult(
        vehicles=[
            DetectedVehicle(
                vehicle_type="car",
                box=(10, 10, 80, 80),
                crop=dummy_crop,
                crop_box=(10, 10, 80, 80),
            )
        ],
        humans_outside=0,
        humans_inside=0,
    )

    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = [{"plate": "RJ09GA0165", "state": "Rajasthan"}]
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="car.jpg")
    called_img = mock_ocr.recognize.call_args[0][0]
    assert isinstance(called_img, Image.Image)
    assert called_img.size == (70, 70)
    assert response.results[0].plate == "RJ09GA0165"
    assert response.results[0].vehicle_type == "car"


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_no_vehicle_detected_full_frame(mock_yolo, mock_ocr_cls, sample_image_bytes):
    mock_yolo.return_value = DetectionResult(
        vehicles=[],
        humans_outside=0,
        humans_inside=0,
    )

    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = [{"plate": "DL01AB1234", "state": "Delhi"}]
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="plate_crop.jpg")
    assert len(response.results) == 1
    assert response.results[0].plate == "DL01AB1234"
    assert response.results[0].vehicle_type is None


def test_resolve_bytes_paths_and_errors(tmp_path, sample_image_bytes):
    import pytest

    from app.core.exceptions import InvalidImageError
    from app.services.pipeline import _resolve_bytes

    valid_file = tmp_path / "valid.jpg"
    valid_file.write_bytes(sample_image_bytes)
    assert _resolve_bytes(str(valid_file)) == sample_image_bytes

    with pytest.raises(InvalidImageError, match="Failed to read image file"):
        _resolve_bytes(str(tmp_path / "does_not_exist.jpg"))

    from typing import Any, cast

    with pytest.raises(InvalidImageError, match="Unsupported image input type"):
        _resolve_bytes(cast(Any, 12345))


def test_validate_plate_results():
    from app.services.pipeline import validate_plate_results

    assert validate_plate_results(None) == []
    assert validate_plate_results("invalid") == []

    results = validate_plate_results([{"plate": 123}, {"plate": "DL01AB1234", "state": "Delhi"}])
    assert len(results) == 1
    assert results[0].plate == "DL01AB1234"


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_ocr_crop_fallback_and_error_handling(mock_yolo, mock_ocr_cls, sample_image_bytes):
    from app.core.exceptions import ANPRServiceError

    dummy_crop = Image.new("RGB", (50, 50))
    detection = DetectionResult(
        vehicles=[
            DetectedVehicle(
                vehicle_type="truck",
                box=(0, 0, 50, 50),
                crop=dummy_crop,
                crop_box=(0, 0, 50, 50),
            )
        ],
        humans_outside=0,
        humans_inside=0,
    )
    mock_yolo.return_value = detection

    mock_ocr = MagicMock()
    mock_ocr.recognize.side_effect = [
        [{"plate": "N/A", "state": "N/A"}],
        [{"plate": "MH12AB1234", "state": "Maharashtra"}],
    ]
    mock_ocr_cls.return_value = mock_ocr

    resp = recognize_plate_image(sample_image_bytes, filename="fallback.jpg")
    assert mock_ocr.recognize.call_count == 2
    assert resp.results[0].plate == "MH12AB1234"
    assert resp.results[0].vehicle_type == "truck"

    # OCR raises ANPRServiceError
    mock_ocr.recognize.side_effect = ANPRServiceError("OCR model crashed")
    resp_err = recognize_plate_image(sample_image_bytes, filename="error.jpg")
    assert resp_err.results == []


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_unread_vehicle_plate_is_none(mock_yolo, mock_ocr_cls, sample_image_bytes):
    dummy_crop = Image.new("RGB", (50, 50))
    detection = DetectionResult(
        vehicles=[
            DetectedVehicle(
                vehicle_type="car",
                box=(0, 0, 50, 50),
                crop=dummy_crop,
                crop_box=(0, 0, 50, 50),
            )
        ],
        humans_outside=0,
        humans_inside=0,
    )
    mock_yolo.return_value = detection

    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = []
    mock_ocr_cls.return_value = mock_ocr

    resp = recognize_plate_image(sample_image_bytes, filename="no_plate.jpg")
    assert len(resp.results) == 1
    assert resp.results[0].plate is None
    assert resp.results[0].vehicle_type == "car"


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_no_vehicle_detected_multiple_plates_fallback(mock_yolo, mock_ocr_cls, sample_image_bytes):
    mock_yolo.return_value = DetectionResult(
        vehicles=[],
        humans_outside=0,
        humans_inside=0,
    )

    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = [
        {"plate": "DL01AB1234", "state": "Delhi", "box": (10, 10, 50, 30)},
        {"plate": "MH12AB1234", "state": "Maharashtra", "box": (100, 100, 150, 130)},
        {"plate": "RJ09GA0165", "state": "Rajasthan", "box": (200, 200, 260, 230)},
    ]
    mock_ocr_cls.return_value = mock_ocr

    response = recognize_plate_image(sample_image_bytes, filename="multiple_plates_fallback.jpg")
    assert len(response.results) == 3
    assert [r.plate for r in response.results] == ["DL01AB1234", "MH12AB1234", "RJ09GA0165"]
    assert all(r.vehicle_type is None for r in response.results)


@patch("app.services.pipeline.PlateRecognizer")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_recognize_single_vehicle_multiple_plates(mock_yolo, mock_ocr_cls, sample_image_bytes):
    dummy_crop = Image.new("RGB", (100, 100))
    detection = DetectionResult(
        vehicles=[
            DetectedVehicle(
                vehicle_type="truck",
                box=(10, 10, 100, 100),
                crop=dummy_crop,
                crop_box=(10, 10, 100, 100),
            )
        ],
        humans_outside=0,
        humans_inside=0,
    )
    mock_yolo.return_value = detection

    mock_ocr = MagicMock()
    mock_ocr.recognize.return_value = [
        {"plate": "MH12AB1234", "state": "Maharashtra", "box": (5, 5, 45, 25)},
        {"plate": "MH12CD5678", "state": "Maharashtra", "box": (5, 50, 45, 70)},
    ]
    mock_ocr_cls.return_value = mock_ocr

    resp = recognize_plate_image(sample_image_bytes, filename="truck_multi_plate.jpg")
    assert len(resp.results) == 2
    assert [r.plate for r in resp.results] == ["MH12AB1234", "MH12CD5678"]
    assert all(r.vehicle_type == "truck" for r in resp.results)
    assert resp.results[0].box == (15, 15, 55, 35)
    assert resp.results[1].box == (15, 60, 55, 80)


