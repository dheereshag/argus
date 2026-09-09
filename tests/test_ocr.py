"""Dedicated coverage for PlateRecognizer OCR-token parsing and plate extraction logic."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.image_processing import load_rgb
from app.services.ocr import PlateRecognizer


def _mock_engine(txts, scores, boxes):
    engine = MagicMock()
    engine.return_value = SimpleNamespace(txts=txts, scores=scores, boxes=boxes)
    return engine


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_single_token_exact_plate_match(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09GA0165"],
        scores=[0.95],
        boxes=[[0, 0, 100, 30]],
    )
    recognizer = PlateRecognizer()
    result = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "RJ09GA0165"
    assert result[0]["state"] == "Rajasthan"


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_two_line_spatial_pairing(mock_get_engine, sample_image_bytes):
    # Plate split across two OCR lines, e.g. state+district on one line,
    # series+serial directly below it.
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09", "GA0165"],
        scores=[0.90, 0.92],
        boxes=[[0, 0, 40, 20], [0, 22, 60, 42]],
    )
    recognizer = PlateRecognizer()
    result = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "RJ09GA0165"


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_decal_words_filtered_no_false_match(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["ASHOK LEYLAND", "TRANSPORT"],
        scores=[0.90, 0.90],
        boxes=[[0, 0, 100, 20], [0, 22, 100, 42]],
    )
    recognizer = PlateRecognizer()
    result = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "N/A"


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_no_ocr_tokens_returns_empty(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(txts=[], scores=[], boxes=[])
    recognizer = PlateRecognizer()
    result = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result == []


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_low_confidence_token_discarded(mock_get_engine, sample_image_bytes):
    mock_get_engine.return_value = _mock_engine(
        txts=["RJ09GA0165"],
        scores=[0.05],  # below the 0.20 score floor
        boxes=[[0, 0, 100, 30]],
    )
    recognizer = PlateRecognizer()
    result = recognizer._extract_plates_from_image_array(load_rgb(sample_image_bytes))

    assert result[0]["plate"] == "N/A"


def test_get_box_centroid_and_geometry():
    recognizer = PlateRecognizer()
    # 4-point polygon
    poly = [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
    cx, cy = recognizer._get_box_centroid(poly)
    assert cx == 30.0
    assert cy == 30.0

    # 4-element box
    box4 = [10.0, 20.0, 50.0, 40.0]
    cx4, cy4 = recognizer._get_box_centroid(box4)
    assert cx4 == 30.0
    assert cy4 == 30.0

    # Invalid box
    assert recognizer._get_box_centroid(None) == (None, None)
    assert recognizer._get_box_centroid(["a", "b", "c", "d"]) == (None, None)


def test_recognizer_parse_plate_info_delegation():
    recognizer = PlateRecognizer()
    res = recognizer.parse_plate_info("MH12AB1234")
    assert res is not None
    assert res["plate"] == "MH12AB1234"


@patch("app.services.ocr.PlateRecognizer.get_engine")
def test_ocr_inference_engine_exception(mock_get_engine, sample_image_bytes):
    mock_get_engine.side_effect = RuntimeError("Engine init crashed")
    recognizer = PlateRecognizer()
    tokens = recognizer._run_ocr_inference(load_rgb(sample_image_bytes))
    assert tokens == []


@patch("app.services.ocr.PlateRecognizer._extract_plates_from_image_array")
def test_recognize_two_pass_enhancement_success(mock_extract, sample_image_bytes):
    mock_extract.side_effect = [
        [{"plate": "N/A", "state": "N/A"}],
        [{"plate": "RJ09GA0165", "state": "Rajasthan"}],
    ]
    recognizer = PlateRecognizer()
    result = recognizer.recognize(sample_image_bytes)
    assert result[0]["plate"] == "RJ09GA0165"
    assert mock_extract.call_count == 2


@patch("app.services.ocr.PlateRecognizer._enhance_contrast")
@patch("app.services.ocr.PlateRecognizer._extract_plates_from_image_array")
def test_recognize_two_pass_enhancement_error_handled(mock_extract, mock_enhance, sample_image_bytes):
    mock_extract.return_value = [{"plate": "N/A", "state": "N/A"}]
    mock_enhance.side_effect = RuntimeError("Contrast filter error")

    recognizer = PlateRecognizer()
    result = recognizer.recognize(sample_image_bytes)
    assert result[0]["plate"] == "N/A"


def test_ocr_geometry_and_clustering_edge_cases():
    from app.services.ocr import OCRToken

    # 1. Invalid nested box geometry -> raises TypeError/ValueError internally and caught
    cx, cy, box = PlateRecognizer._get_box_geometry(
        [["not_a_number", "bad"], [0, 0], [1, 1], [2, 2]]
    )
    assert cx is None and cy is None and box is None


    # 2. _is_same_horizontal_line when box is None but cy is available
    t1 = OCRToken(text="MH", score=0.9, cx=10.0, cy=20.0, box=None)
    t2 = OCRToken(text="12", score=0.9, cx=40.0, cy=22.0, box=None)
    assert PlateRecognizer._is_same_horizontal_line(t1, t2) is True

    t3 = OCRToken(text="99", score=0.9, cx=40.0, cy=50.0, box=None)
    assert PlateRecognizer._is_same_horizontal_line(t1, t3) is False

    t4 = OCRToken(text="00", score=0.9, cx=40.0, cy=None, box=None)
    assert PlateRecognizer._is_same_horizontal_line(t1, t4) is False

    # 3. _build_spatial_pairs when cx/cy are None
    tok_a = OCRToken(text="MH12", score=0.9, cx=None, cy=None, box=None)
    tok_b = OCRToken(text="AB1234", score=0.9, cx=None, cy=None, box=None)
    pairs = PlateRecognizer._build_spatial_pairs([tok_a, tok_b])
    assert len(pairs) == 2

    # 4. _compute_token_bounds with no valid boxes
    assert PlateRecognizer._compute_token_bounds([tok_a, tok_b]) is None

    # 5. _collect_candidates when parse_plate_info returns None
    with patch("app.services.ocr.parse_plate_info", return_value=None):
        tok_valid = OCRToken(text="MH12AB1234", score=0.95, cx=10.0, cy=10.0, box=(0, 0, 10, 10))
        cands = PlateRecognizer()._collect_candidates([tok_valid], [], [], "MH12AB1234")
        assert cands == []



@patch("app.services.ocr.PlateRecognizer._enhance_contrast")
@patch("app.services.ocr.PlateRecognizer._extract_plates_from_image_array")
def test_recognize_two_pass_enhancement_both_na(mock_extract, mock_enhance, sample_image_bytes):
    mock_extract.side_effect = [
        [{"plate": "N/A", "state": "N/A"}],
        [{"plate": "N/A", "state": "N/A"}],
    ]
    recognizer = PlateRecognizer()
    result = recognizer.recognize(sample_image_bytes)
    assert result[0]["plate"] == "N/A"

