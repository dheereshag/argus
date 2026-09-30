"""Tests for the 4-tier cascaded Fast-ALPR and RapidOCR ANPR fallback pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.schemas import DetectedVehicle, DetectionResult
from app.services.ocr.fast_alpr_padding import compute_padded_crop
from app.services.pipeline.orchestrator import recognize_plate_image


def test_fast_alpr_padding_expansion():
    """Verify compute_padded_crop correctly applies 15% X and 10% Y margin expansion."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = MagicMock(x1=20, y1=20, x2=60, y2=40)
    crop = compute_padded_crop(dummy_img, bbox, pad_x_pct=0.15, pad_y_pct=0.10)
    # Box width 40 -> pad_x = max(10, 6) = 10 -> x1=10, x2=70 (width=60)
    # Box height 20 -> pad_y = max(5, 2) = 5 -> y1=15, y2=45 (height=30)
    assert crop.shape[0] == 30
    assert crop.shape[1] == 60


@patch("app.services.pipeline.orchestrator.run_fast_alpr_pipeline")
@patch("app.services.pipeline.VehicleDetector.detect")
def test_cascade_tier1_fast_alpr_full_frame_early_exit(mock_yolo, mock_fast_full, sample_image_bytes):
    """Tier 1: When Fast-ALPR finds plate on full frame, vehicle detection is skipped."""
    mock_fast_full.return_value = ["RJ09GA0165"]

    resp = recognize_plate_image(sample_image_bytes, filename="clear_plate.jpg")

    assert mock_fast_full.call_count == 1
    assert mock_yolo.call_count == 0
    assert len(resp.results) == 1
    assert resp.results[0].plate == "RJ09GA0165"
    assert resp.results[0].execution_time_ms > 0
    assert resp.execution_time_ms > 0


@patch("app.services.pipeline.orchestrator._run_rapidocr_fallback")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_on_crops")
@patch("app.services.pipeline.VehicleDetector.detect")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_pipeline")
def test_cascade_tier2_fast_alpr_crop_early_exit(
    mock_fast_full, mock_yolo, mock_fast_crop, mock_rapid, sample_image_bytes
):
    """Tier 2: When Tier 1 is empty, Fast-ALPR on vehicle crop succeeds and RapidOCR is skipped."""
    mock_fast_full.return_value = []
    dummy_crop = Image.new("RGB", (60, 60))
    mock_yolo.return_value = DetectionResult(
        vehicles=[DetectedVehicle(vehicle_type="car", box=(10, 10, 50, 50), crop=dummy_crop)]
    )
    mock_fast_crop.return_value = ["MH12AB1234"]

    resp = recognize_plate_image(sample_image_bytes, filename="distant_vehicle.jpg")

    assert mock_fast_full.call_count == 1
    assert mock_yolo.call_count == 1
    assert mock_fast_crop.call_count == 1
    assert mock_rapid.call_count == 0
    assert len(resp.results) == 1
    assert resp.results[0].plate == "MH12AB1234"
    assert resp.results[0].execution_time_ms > 0


@patch("app.services.pipeline.orchestrator._run_rapidocr_fallback")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_on_crops")
@patch("app.services.pipeline.VehicleDetector.detect")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_pipeline")
def test_cascade_tier3_rapidocr_fallback(
    mock_fast_full, mock_yolo, mock_fast_crop, mock_rapid, sample_image_bytes
):
    """Tier 3/4: When Fast-ALPR yields no plates, RapidOCR fallback is executed."""
    mock_fast_full.return_value = []
    dummy_crop = Image.new("RGB", (60, 60))
    mock_yolo.return_value = DetectionResult(
        vehicles=[DetectedVehicle(vehicle_type="truck", box=(10, 10, 50, 50), crop=dummy_crop)]
    )
    mock_fast_crop.return_value = []
    mock_rapid.return_value = ["DL01AB1234"]

    resp = recognize_plate_image(sample_image_bytes, filename="difficult_plate.jpg")

    assert mock_fast_full.call_count == 1
    assert mock_fast_crop.call_count == 1
    assert mock_rapid.call_count == 1
    assert len(resp.results) == 1
    assert resp.results[0].plate == "DL01AB1234"
    assert resp.results[0].execution_time_ms > 0


@patch("app.services.pipeline.orchestrator._run_rapidocr_fallback")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_on_crops")
@patch("app.services.pipeline.VehicleDetector.detect")
@patch("app.services.pipeline.orchestrator.run_fast_alpr_pipeline")
def test_cascade_all_tiers_empty_returns_clean_response(
    mock_fast_full, mock_yolo, mock_fast_crop, mock_rapid, sample_image_bytes
):
    """When all 4 tiers produce no valid plates, thin response has results=[]."""
    mock_fast_full.return_value = []
    mock_yolo.return_value = DetectionResult(vehicles=[])
    mock_fast_crop.return_value = []
    mock_rapid.return_value = []

    resp = recognize_plate_image(sample_image_bytes, filename="empty_road.jpg")

    assert resp.results == []
    assert resp.execution_time_ms > 0


def test_plate_resolver_preserves_valid_gt_series():
    """Verify that legitimate series like GT in RJ14GT4976 are not mutated to GJ."""
    from app.services.plate_rules import resolve_plate_from_text

    assert resolve_plate_from_text("RJ14GT4976") == "RJ14GT4976"


def test_plate_resolver_corrects_g3_digit_misread():
    """Verify that digit-in-series OCR error G3 and prefix RT are corrected to RJ14GJ4976."""
    from app.services.plate_rules import resolve_plate_from_text

    assert resolve_plate_from_text("RT14G34976") == "RJ14GJ4976"


def test_plate_resolver_filters_decals_and_phones():
    """Verify that commercial decal words and driver phone numbers are rejected."""
    from app.services.plate_rules import resolve_plate_from_text

    assert resolve_plate_from_text("FASTAG") is None
    assert resolve_plate_from_text("GOODSCARRIER") is None
    assert resolve_plate_from_text("9414378858") is None
    assert resolve_plate_from_text("PASSING") is None
