"""
Unit and integration tests for Indian ANPR domain improvements.

Verifies:
  - 11-character plate normalization (3-letter series like DL01CAA1234, MH02CRB1234).
  - HSRP 'IND' watermark and blue strip prefix stripping.
  - Multi-token horizontal line clustering and 2-line stacked plate reconstruction.
  - Telangana 'TG' state code resolution (2024 gazette standard).
  - 10-digit mobile telephone number filtering and expanded decal blacklisting.
  - OCR confidence and bounding box coordinate propagation in PlateResult.
  - Bharat (BH) series character confusion resilience (e.g. 228H -> 22BH).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from app.constants import STATE_CODES
from app.schemas import PlateResult
from app.services.image_processing import load_rgb
from app.services.ocr import PlateRecognizer
from app.services.plate_rules import (
    is_decal_word,
    is_phone_number,
    normalize_candidate_strings,
    parse_plate_info,
)


def _mock_ocr_engine(txts: list[str], scores: list[float], boxes: Any):
    engine = MagicMock()
    engine.return_value = SimpleNamespace(txts=txts, scores=scores, boxes=boxes)
    return engine


# ------------------------------------------------------------------------------
# 1. State Codes & 2024 Updates
# ------------------------------------------------------------------------------


def test_tg_telangana_state_code():
    """Verify that 2024 Telangana state prefix 'TG' is supported."""
    assert "TG" in STATE_CODES
    assert STATE_CODES["TG"] == "Telangana"

    parsed = parse_plate_info("TG09AB1234")
    assert parsed is not None
    assert parsed["plate"] == "TG09AB1234"
    assert parsed["state"] == "Telangana"


# ------------------------------------------------------------------------------
# 2. 11-Character Plate Normalization (3-Letter Series)
# ------------------------------------------------------------------------------


def test_11_character_plate_normalization():
    """11-character plates with 3-letter series must undergo character disambiguation."""
    # DL 01 CAA 1234 (11 chars) with OCR errors: 'O' in district, '8' in number
    raw = "DLO1CAA1238"
    normalized = normalize_candidate_strings(raw)
    assert "DL01CAA1238" in normalized

    # Direct parse of valid 11-char plate
    info = parse_plate_info("DL01CAA1234")
    assert info is not None
    assert info["plate"] == "DL01CAA1234"
    assert info["state"] == "Delhi"

    # Maharashtra 11-char plate
    info_mh = parse_plate_info("MH02CRB1234")
    assert info_mh is not None
    assert info_mh["plate"] == "MH02CRB1234"
    assert info_mh["state"] == "Maharashtra"


# ------------------------------------------------------------------------------
# 3. HSRP 'IND' Strip Prefix Stripping
# ------------------------------------------------------------------------------


def test_hsrp_ind_prefix_stripping():
    """HSRP plates with leading 'IND' national strip must be stripped and recognized."""
    raw = "INDMH12AB1234"
    normalized = normalize_candidate_strings(raw)
    assert "MH12AB1234" in normalized

    info = parse_plate_info(raw)
    assert info is not None
    assert info["plate"] == "MH12AB1234"
    assert info["state"] == "Maharashtra"


# ------------------------------------------------------------------------------
# 4. Bharat (BH) Series OCR Resilience
# ------------------------------------------------------------------------------


def test_bh_series_ocr_confusion_correction():
    """Bharat series with OCR misread '8H' instead of 'BH' should normalize."""
    raw = "228H1234AA"
    normalized = normalize_candidate_strings(raw)
    assert "22BH1234AA" in normalized


# ------------------------------------------------------------------------------
# 5. Mobile Phone Number & Commercial Decal Filtering
# ------------------------------------------------------------------------------


def test_phone_number_filtering():
    """10-digit Indian mobile numbers must be filtered out as decals."""
    assert is_phone_number("9414378858") is True
    assert is_phone_number("9845214173") is True
    assert is_decal_word("9414378858") is True
    assert is_decal_word("FASTAG") is True
    assert is_decal_word("PASSING") is True
    assert is_decal_word("IND") is True


# ------------------------------------------------------------------------------
# 6. Multi-Token Horizontal Line & 2-Line Stacked Plate Reconstruction
# ------------------------------------------------------------------------------


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_multi_token_2_line_plate_reconstruction(mock_get_engine, sample_image_bytes):
    """Plate split into 4 tokens across 2 lines (RJ 09 / GA 0165) must be reconstructed."""
    # Line 1: 'RJ' (box: x=10..30, y=10..25) and '09' (box: x=35..55, y=10..25)
    # Line 2: 'GA' (box: x=10..30, y=30..45) and '0165' (box: x=35..75, y=30..45)
    mock_get_engine.return_value = _mock_ocr_engine(
        txts=["RJ", "09", "GA", "0165"],
        scores=[0.95, 0.96, 0.98, 0.97],
        boxes=[
            [[10.0, 10.0], [30.0, 10.0], [30.0, 25.0], [10.0, 25.0]],
            [[35.0, 10.0], [55.0, 10.0], [55.0, 25.0], [35.0, 25.0]],
            [[10.0, 30.0], [30.0, 30.0], [30.0, 45.0], [10.0, 45.0]],
            [[35.0, 30.0], [75.0, 30.0], [75.0, 45.0], [35.0, 45.0]],
        ],
    )
    recognizer = PlateRecognizer()
    results = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert len(results) == 1
    assert results[0]["plate"] == "RJ09GA0165"
    assert results[0]["state"] == "Rajasthan"
    assert results[0]["confidence"] is not None
    assert results[0]["confidence"] > 0.90
    assert results[0]["box"] == (10, 10, 75, 45)


# ------------------------------------------------------------------------------
# 7. Metadata Propagation in PlateResult Schema
# ------------------------------------------------------------------------------


def test_plate_result_metadata_serialization():
    """Verify that PlateResult correctly serializes confidence score and bounding box."""
    data = {
        "plate": "RJ09GA0165",
        "state": "Rajasthan",
        "raw_text": "RJ09 GA0165",
        "confidence": 0.965,
        "box": (10, 10, 75, 45),
    }
    result = PlateResult.model_validate(data)
    assert result.plate == "RJ09GA0165"
    assert result.confidence == 0.965
    assert result.box == (10, 10, 75, 45)

    dump = result.model_dump()
    assert dump["confidence"] == 0.965
    assert dump["box"] == (10, 10, 75, 45)
