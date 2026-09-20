"""
Tests for ANPR system improvements:
  - Cab occupant / vehicle interior containment filtering.
  - Coordinate translation from crop coordinates to full frame space.
  - Expanded 9-character and 8-character Indian plate normalization.
  - State prefix correction in parse_plate_info.
  - Bounded CLAHE contrast enhancement performance invariants.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.config import settings
from app.schemas import RecognitionStatusEnum
from app.services.detector import VehicleDetector
from app.services.ocr import PlateRecognizer
from app.services.pipeline import _adjust_crop_coordinates
from app.services.plate_rules import (
    _normalize_8_char,
    _normalize_9_char,
    normalize_candidate_strings,
    parse_plate_info,
)


def test_is_contained_geometric_evaluations():
    """Verify BoundingBox containment helper under various geometric configurations."""
    detector = VehicleDetector()

    # Inner box fully inside outer box
    inner = (30, 30, 50, 50)
    outer = (10, 10, 100, 100)
    assert detector._is_contained(inner, outer, threshold=0.80) is True

    # Inner box partially overlapping (around 25% overlap)
    partial_inner = (5, 5, 25, 25)  # area = 400, overlap with outer (10,10,100,100) is (10,10,25,25) area = 225 -> 56%
    assert detector._is_contained(partial_inner, outer, threshold=0.80) is False

    # Non-overlapping boxes
    disjoint_inner = (150, 150, 200, 200)
    assert detector._is_contained(disjoint_inner, outer, threshold=0.80) is False

    # Degenerate zero-area boxes
    zero_box = (10, 10, 10, 10)
    assert detector._is_contained(zero_box, outer, threshold=0.80) is False


@patch("app.services.detector.VehicleDetector.get_model")
def test_cab_occupant_ignored_when_policy_enabled(mock_get_model, sample_image_bytes):
    """When ALLOW_CAB_OCCUPANTS is True, a person box fully inside vehicle is not a ground pedestrian."""
    mock_box = MagicMock()
    mock_box.__len__.return_value = 2
    mock_box.cls.cpu().numpy.return_value = [0, 7]  # person (0), truck (7)
    mock_box.conf.cpu().numpy.return_value = [0.85, 0.90]
    # Person (30..50, 30..50) is fully inside truck (10..90, 10..90)
    mock_box.xyxy.cpu().numpy.return_value = [[30, 30, 50, 50], [10, 10, 90, 90]]

    mock_res = MagicMock()
    mock_res.boxes = mock_box
    mock_model = MagicMock()
    mock_model.return_value = [mock_res]
    mock_get_model.return_value = mock_model

    detector = VehicleDetector()
    res = detector.detect(sample_image_bytes)

    assert res.is_eligible is True
    assert res.status is None
    assert res.human_count == 0
    assert len(res.vehicles) == 1


@patch("app.services.detector.VehicleDetector.get_model")
def test_cab_occupant_counted_when_policy_disabled(mock_get_model, sample_image_bytes, monkeypatch: pytest.MonkeyPatch):
    """When ALLOW_CAB_OCCUPANTS is False, a person box inside vehicle is counted."""
    monkeypatch.setattr(settings, "ALLOW_CAB_OCCUPANTS", False)

    mock_box = MagicMock()
    mock_box.__len__.return_value = 2
    mock_box.cls.cpu().numpy.return_value = [0, 7]
    mock_box.conf.cpu().numpy.return_value = [0.85, 0.90]
    mock_box.xyxy.cpu().numpy.return_value = [[30, 30, 50, 50], [10, 10, 90, 90]]

    mock_res = MagicMock()
    mock_res.boxes = mock_box
    mock_model = MagicMock()
    mock_model.return_value = [mock_res]
    mock_get_model.return_value = mock_model

    detector = VehicleDetector()
    res = detector.detect(sample_image_bytes)

    assert res.is_eligible is False
    assert res.status == RecognitionStatusEnum.REJECTED_HUMAN_DETECTED
    assert res.human_count == 1


def test_crop_coordinate_adjustment():
    """Verify bounding box translation from crop-relative coordinates to full-frame space."""
    raw_results = [
        {"plate": "RJ09GA0165", "box": (10, 20, 80, 50)},
        {"plate": "N/A", "box": None},
    ]
    crop_box = (100, 200, 500, 600)

    adjusted = _adjust_crop_coordinates(raw_results, crop_box)

    assert adjusted[0]["box"] == (110, 220, 180, 250)
    assert adjusted[1]["box"] is None

    # No crop box leaves results unchanged
    unchanged = _adjust_crop_coordinates(raw_results, None)
    assert unchanged[0]["box"] == (10, 20, 80, 50)


def test_expanded_9_and_8_char_normalizations():
    """Verify 3-digit registration permutations and series disambiguation in 9 and 8-char plates."""
    # 9-char plate with 3-digit registration number: RJ 09 GA 165
    res_9 = _normalize_9_char("RJ09GA165", "RJ")
    assert any("RJ09GA165" in cand for cand in res_9)

    # 9-char plate with series 'I' misread: RJ 09 IA 165 -> RJ 09 JA 165
    res_9_series = _normalize_9_char("RJ09IA165", "RJ")
    assert any("JA" in cand for cand in res_9_series)

    # 8-char plate with single-digit district and 3-digit number: DL 1 CX 274
    res_8 = _normalize_8_char("DL1CX274", "DL")
    assert any("DL1CX274" in cand for cand in res_8)

    # Full candidate normalization pipeline for 9-char plate
    all_cands = normalize_candidate_strings("RJ09GA165")
    assert "RJ09GA165" in all_cands


def test_state_prefix_corrections_in_parse_plate_info():
    """Verify that parse_plate_info fixes common OCR state prefix confusions."""
    # 0D -> OD (Odisha)
    info_od = parse_plate_info("0D01AB1234")
    assert info_od is not None
    assert info_od["plate"] == "OD01AB1234"
    assert info_od["state"] == "Odisha"

    # 7G -> TG (Telangana)
    info_tg = parse_plate_info("7G01AB1234")
    assert info_tg is not None
    assert info_tg["plate"] == "TG01AB1234"
    assert info_tg["state"] == "Telangana"

    # TC -> TG (Telangana)
    info_tc = parse_plate_info("TC01AB1234")
    assert info_tc is not None
    assert info_tc["plate"] == "TG01AB1234"
    assert info_tc["state"] == "Telangana"


def test_bounded_clahe_contrast_enhancement():
    """Verify that _enhance_contrast bounds upscaling and does not explode large images."""
    recognizer = PlateRecognizer()

    # Small image (100x100) -> scaled up, bounded by 640
    small_img = Image.new("RGB", (100, 100), color="gray")
    enhanced_small = recognizer._enhance_contrast(small_img)
    assert enhanced_small.width > 100
    assert enhanced_small.width <= 640

    # Large image (500x500) -> min edge >= 300, not scaled up
    large_img = Image.new("RGB", (500, 500), color="gray")
    enhanced_large = recognizer._enhance_contrast(large_img)
    assert enhanced_large.size == (500, 500)
