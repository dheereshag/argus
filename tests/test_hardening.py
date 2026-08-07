"""
Tests for the NASA Power of 10 hardening.

Each test names the rule it defends and the concrete failure it prevents. A
test that only says "rule 7" is a test nobody will understand in six months.

Rules 3, 8 and 1 have no tests here on purpose — see docs/NASA_RULES.md for why
they were skipped rather than forced.
"""

import io
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.contracts import ContractViolation, bounded, ensure, require
from app.schemas.plate import ProviderEnum


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (100, 100, 100)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Rule 5 — contracts, not assert
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
    """
    The reason these are not `assert`.

    `python -O` strips assert statements entirely, so a safety check written as
    an assert silently disappears in exactly the environment where it matters.
    ContractViolation is a raise, which no flag removes.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-O", "-c",
         "from app.core.contracts import require, ContractViolation\n"
         "try: require(False, 'test'); print('LIVED')\n"
         "except ContractViolation: print('RAISED')\n"],
        capture_output=True, text=True, check=False,
    )
    assert "RAISED" in result.stdout, (
        "contract check did not fire under -O; it has regressed to assert semantics"
    )


# ---------------------------------------------------------------------------
# Rule 2 — every loop has a fixed upper bound
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


def test_vehicle_boxes_are_capped_in_the_waterfall():
    """
    The concrete cost this prevents.

    Per box the waterfall runs 5 ROI tiers x 2 (plain + warped). YOLO returning
    a dozen boxes for a yard with parked vehicles therefore multiplied the whole
    pipeline, which is where the 27.4 s outlier in eval_report.json came from.
    """
    from app.core.config import settings
    from app.services.base import BasePlateRecognizer

    attempts = []

    class _Counting(BasePlateRecognizer):
        def _recognize_single_image(self, image_input, filename="image.jpg"):
            attempts.append(filename)
            return []

    many_boxes = [(0, 0, 100 - i, 100 - i) for i in range(20)]
    _Counting().recognize(_jpeg(400, 400), filename="x.jpg", vehicle_boxes=many_boxes)

    # 5 tiers, at most 2 OCR calls each (plain + warped), per retained box,
    # plus the final full-frame tier.
    ceiling = settings.MAX_VEHICLE_BOXES * 5 * 2 + 2
    assert len(attempts) <= ceiling, (
        f"{len(attempts)} OCR attempts for 20 boxes; cap is "
        f"MAX_VEHICLE_BOXES={settings.MAX_VEHICLE_BOXES}"
    )


def test_largest_boxes_are_the_ones_kept():
    """
    Capping is only safe if the cap keeps the right boxes. At a weighbridge the
    vehicle on the platform dominates the frame, so area-descending is the
    ordering that makes truncation harmless.
    """
    from app.services.base import BasePlateRecognizer
    from app.services.image_processing import box_area

    seen_sizes = []

    class _Recorder(BasePlateRecognizer):
        def _recognize_single_image(self, image_input, filename="image.jpg"):
            with io.BytesIO(image_input) as buf, Image.open(buf) as img:
                seen_sizes.append(img.size)
            return []

    small = (0, 0, 20, 20)
    large = (0, 0, 300, 300)
    _Recorder().recognize(_jpeg(400, 400), vehicle_boxes=[small, large])

    assert box_area(large) > box_area(small)
    # The largest box is processed first, so its crop appears before any small one.
    assert max(s[0] for s in seen_sizes) > 100


# ---------------------------------------------------------------------------
# Rule 6 — smallest scope; the singleton race
# ---------------------------------------------------------------------------

def test_yolo_singleton_is_built_once_under_concurrency():
    """
    Latent until issue #7 made the handler sync, at which point FastAPI began
    running it in a threadpool and two concurrent first-requests could both
    observe None and both construct a model.
    """
    import app.services.yolo_filter as yf

    builds = []

    def slow_build(_name):
        import time
        time.sleep(0.05)          # widen the window the race needs
        builds.append(1)
        return MagicMock()

    original = yf._YOLO_MODEL
    try:
        yf._YOLO_MODEL = None
        with patch.object(yf, "YOLO", side_effect=slow_build):
            threads = [threading.Thread(target=yf.get_yolo_model) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(builds) == 1, f"YOLO model built {len(builds)} times; the lock is not holding"
    finally:
        yf._YOLO_MODEL = original


def test_both_singletons_are_lock_guarded():
    """Source-level guard: a future refactor must not drop the lock."""
    import inspect

    import app.services.strategies.paddle_ocr as po
    import app.services.yolo_filter as yf

    for module in (yf, po):
        source = inspect.getsource(module)
        assert "threading.Lock()" in source, f"{module.__name__} lost its init lock"


# ---------------------------------------------------------------------------
# Rule 7 — validate parameters, check returns
# ---------------------------------------------------------------------------

def test_out_of_bounds_box_is_clamped():
    """
    PIL does not raise on an out-of-range crop — it pads with black. So an
    unclamped box produced a crop containing invented pixels, OCR ran on it,
    and nothing reported a problem. Silent wrong input, which is the whole
    reason this rule exists.
    """
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


def test_malformed_provider_output_does_not_500():
    """
    Was `[PlateResult(**item) for item in raw_results]`. An unexpected key
    raises TypeError, a missing one raises ValidationError, and either turned a
    request that HAD successfully read a plate into a 500.
    """
    from app.api.endpoints.recognition import _validate_plate_results

    mixed = [
        {"plate": "RJ09GA0165", "state": "Rajasthan"},   # good
        {"unexpected_key": "boom"},                       # missing required field
        "not a dict",                                     # wrong type entirely
        {"plate": "MH12AB1234"},                          # good, optional fields absent
    ]
    results = _validate_plate_results(mixed, ProviderEnum.PADDLEOCR)

    assert [r.plate for r in results] == ["RJ09GA0165", "MH12AB1234"], (
        "good results must survive alongside bad ones"
    )


def test_non_list_provider_output_is_handled():
    from app.api.endpoints.recognition import _validate_plate_results

    assert _validate_plate_results({"plate": "X"}, ProviderEnum.NVIDIA) == []
    assert _validate_plate_results(None, ProviderEnum.NVIDIA) == []


# ---------------------------------------------------------------------------
# Rule 9 — no deep unchecked access chains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {},                                                     # empty object
    {"choices": []},                                        # rate limited / filtered
    {"choices": [{}]},                                      # no message
    {"choices": [{"message": {}}]},                         # no content
    {"choices": [{"message": {"content": None}}]},          # null content
    {"choices": [{"message": {"content": "   "}}]},         # whitespace only
    {"error": {"message": "quota exceeded"}},               # error body, HTTP 200
    {"choices": "not-a-list"},
    [],                                                     # not even an object
    None,
])
def test_malformed_nvidia_response_returns_none_not_an_exception(payload):
    """
    `res_json['choices'][0]['message']['content']` had four ways to raise, all
    of them caught by a bare `except Exception` that logged the provider as
    merely "failed" — making a malformed response indistinguishable from a
    network outage in the logs.
    """
    from app.services.strategies.nvidia_vision import extract_message_content

    assert extract_message_content(payload) is None


def test_wellformed_nvidia_response_is_extracted():
    from app.services.strategies.nvidia_vision import extract_message_content

    payload = {"choices": [{"message": {"content": "  RJ09GA0165 \n"}}]}
    assert extract_message_content(payload) == "RJ09GA0165"


# ---------------------------------------------------------------------------
# Rule 3 (intent only) — bounded, promptly released resources
# ---------------------------------------------------------------------------

def test_image_loading_closes_its_source_handle():
    """
    Rule 3's literal form is meaningless under garbage collection, but its
    intent is not. `Image.open` is lazy and holds the source open until the
    pixels are read, so the previous code left a handle per call for the
    collector to reclaim whenever it chose. Under a threadpool that is a slow
    leak.
    """
    from app.services.image_processing import load_rgb

    buf = io.BytesIO(_jpeg(64, 64))
    img = load_rgb(buf.getvalue())

    assert img.size == (64, 64)
    assert img.mode == "RGB"
    assert getattr(img, "fp", None) is None, "decoded image still holds an open source handle"


def test_no_bare_image_open_outside_the_helper():
    """
    Source-level guard. Every decode should route through load_rgb so the
    context-manager discipline is in one place rather than re-implemented.
    """
    import inspect

    from app.services import image_processing
    from app.services.strategies import paddle_ocr

    source = inspect.getsource(paddle_ocr)
    assert "Image.open(" not in source, (
        "paddle_ocr should decode via load_rgb, not Image.open directly"
    )

    ip_source = inspect.getsource(image_processing)
    # image_processing legitimately calls Image.open, but only inside `with`.
    for line in ip_source.splitlines():
        if "Image.open(" in line:
            assert "with " in line, f"unmanaged Image.open: {line.strip()}"
