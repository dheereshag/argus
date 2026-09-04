"""
Tests for runtime contracts, bounds, resource lifecycle, and error isolation.
"""

import io
from unittest.mock import MagicMock, patch

import pytest

from app.core.contracts import ContractViolation, bounded, ensure, require
from app.services.pipeline import validate_plate_results
from tests.conftest import create_test_jpeg

# ---------------------------------------------------------------------------
# Runtime contracts (preconditions and postconditions)
# ---------------------------------------------------------------------------


def test_require_and_ensure_raise_on_violation():
    with pytest.raises(ContractViolation, match="precondition"):
        require(False, "box must be non-empty")
    with pytest.raises(ContractViolation, match="postcondition"):
        ensure(False, "crop must be non-empty")


def test_require_passes_on_truthy():
    require(1, "fine")
    ensure("yes", "fine")


def test_contracts_survive_optimised_mode():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            (
                "from app.core.contracts import require, ContractViolation\n"
                "try: require(False, 'test'); print('LIVED')\n"
                "except ContractViolation: print('RAISED')\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "RAISED" in result.stdout, "contract check did not fire under -O; it has regressed to assert semantics"


# ---------------------------------------------------------------------------
# Fixed loop and sequence bounds
# ---------------------------------------------------------------------------


def test_bounded_truncates_and_preserves_order():
    assert bounded(list(range(100)), 3, "things") == [0, 1, 2]


def test_bounded_passes_short_sequences_through():
    assert bounded([1, 2], 5, "things") == [1, 2]
    assert bounded([], 5, "things") == []
    assert bounded(None, 5, "things") == []


def test_bounded_rejects_a_nonsense_limit():
    with pytest.raises(ContractViolation):
        bounded([1, 2, 3], 0, "things")


# ---------------------------------------------------------------------------
# Model singleton initialization
# ---------------------------------------------------------------------------


def test_yolo_singleton_initializes_once():
    import app.services.yolo_filter as yf

    builds = []

    def mock_build(_name):
        builds.append(1)
        return MagicMock()

    original = yf._YOLO_MODEL
    try:
        yf._YOLO_MODEL = None
        with patch.object(yf, "YOLO", side_effect=mock_build):
            m1 = yf.get_yolo_model()
            m2 = yf.get_yolo_model()
        assert m1 is m2
        assert len(builds) == 1
    finally:
        yf._YOLO_MODEL = original


# ---------------------------------------------------------------------------
# Parameter validation and boundary clamping
# ---------------------------------------------------------------------------


def test_out_of_bounds_box_is_clamped():
    from app.services.image_processing import clamp_box

    assert clamp_box((-50, -50, 5000, 5000), 640, 480) == (0, 0, 640, 480)


def test_corner_swapped_box_is_repaired():
    from app.services.image_processing import clamp_box

    assert clamp_box((300, 200, 100, 50), 640, 480) == (100, 50, 300, 200)


@pytest.mark.parametrize("bad", [None, (), (1, 2), (0, 0, 2, 2), "nope", (0, 0, "x", 4)])
def test_unusable_boxes_are_rejected_not_cropped(bad):
    from app.services.image_processing import clamp_box

    assert clamp_box(bad, 640, 480) is None


def test_clamp_rejects_nonsense_image_dimensions():
    from app.services.image_processing import clamp_box

    with pytest.raises(ContractViolation):
        clamp_box((0, 0, 10, 10), 0, 480)


def test_malformed_provider_output_does_not_error():
    mixed = [
        {"plate": "RJ09GA0165", "state": "Rajasthan"},
        {"unexpected_key": "boom"},
        "not a dict",
        {"plate": "MH12AB1234"},
    ]
    results = validate_plate_results(mixed)

    assert [r.plate for r in results] == ["RJ09GA0165", "MH12AB1234"]


def test_non_list_provider_output_is_handled():
    assert validate_plate_results({"plate": "X"}) == []
    assert validate_plate_results(None) == []


# ---------------------------------------------------------------------------
# Bounded and promptly released resources
# ---------------------------------------------------------------------------


def test_image_loading_closes_its_source_handle():
    from app.services.image_processing import load_rgb

    buf = io.BytesIO(create_test_jpeg(64, 64))
    img = load_rgb(buf.getvalue())

    assert img.size == (64, 64)
    assert img.mode == "RGB"
    assert getattr(img, "fp", None) is None


def test_no_bare_image_open_outside_the_helper():
    import inspect

    from app.services import image_processing
    from app.services.strategies import docling_ocr

    source = inspect.getsource(docling_ocr)
    assert "Image.open(" not in source

    ip_source = inspect.getsource(image_processing)
    for line in ip_source.splitlines():
        if "Image.open(" in line:
            assert "with " in line, f"unmanaged Image.open: {line.strip()}"
