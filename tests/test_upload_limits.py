"""
Tests for the four production failure modes fixed in issue #7:
blocking event loop, missing HTTP timeouts, unbounded uploads, decode bombs.

Each of these ends a demo, and none of them were covered by the existing suite —
test_api.py mocks the recogniser wholesale, so the request path that actually
falls over was never exercised.
"""

import inspect
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from app.api.endpoints import recognition
from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.services.image_processing import decode_and_downscale


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# The handler must not block the event loop
# --------------------------------------------------------------------------

def test_recognize_handler_is_sync():
    """
    The handler was `async def` while doing blocking YOLO, PaddleOCR and
    requests.post work, so the service handled one request at a time.

    FastAPI dispatches sync handlers to a threadpool. If someone reintroduces
    `async def` without moving the blocking work off the loop, fail here rather
    than discovering it under load.
    """
    assert not inspect.iscoroutinefunction(recognition.recognize_plate), (
        "recognize_plate must stay a sync `def` — it performs blocking CPU and "
        "socket work that would otherwise stall the event loop."
    )


def test_concurrent_requests_are_served_in_parallel(client, sample_image_bytes, monkeypatch):
    """
    Two requests should overlap. With the old async handler they serialised.
    """
    def slow_filter(image_input, *args, **kwargs):
        time.sleep(0.4)
        return {
            "is_eligible": False,
            "status": "rejected_no_four_wheeler",
            "status_message": "no vehicle",
            "vehicle_detected": False,
            "vehicle_type": None,
            "human_detected": False,
            "vehicle_box": None,
            "vehicle_count": 0,
        }

    monkeypatch.setattr(recognition, "filter_vehicle_and_occupancy", slow_filter)

    def fire():
        return client.post(
            "/recognize",
            files={"file": ("car.jpg", sample_image_bytes, "image/jpeg")},
        )

    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = [f.result() for f in [pool.submit(fire), pool.submit(fire)]]
    elapsed = time.time() - start

    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 0.75, (
        f"Two 0.4s requests took {elapsed:.2f}s — they serialised. "
        "The handler is blocking the event loop."
    )


# --------------------------------------------------------------------------
# Upload size
# --------------------------------------------------------------------------

def test_oversized_upload_is_rejected(client):
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (settings.MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/recognize",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "PayloadTooLargeError"


def test_empty_upload_is_rejected(client):
    response = client.post("/recognize", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Decode bombs and downscaling
# --------------------------------------------------------------------------

def test_pixel_budget_is_enforced(monkeypatch):
    """
    A small file can declare enormous dimensions. Guard on the pixel count from
    the header, before the full decode allocates anything.
    """
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(PayloadTooLargeError):
        decode_and_downscale(_jpeg(200, 200))


def test_large_image_is_downscaled():
    out = decode_and_downscale(_jpeg(4000, 3000))
    assert max(Image.open(io.BytesIO(out)).size) <= settings.MAX_IMAGE_EDGE_PX


def test_downscaled_output_fits_plate_recognizer_ceiling():
    """
    plate_recognizer.py skips any payload >= 3.5 MB, which silently made large
    images unrecognisable. Downscaling must keep output under that ceiling.
    """
    assert len(decode_and_downscale(_jpeg(4000, 3000))) < 3.5 * 1024 * 1024


def test_small_image_is_not_upscaled():
    out = decode_and_downscale(_jpeg(320, 240))
    assert Image.open(io.BytesIO(out)).size == (320, 240)


def test_undecodable_bytes_raise_invalid_image():
    with pytest.raises(InvalidImageError):
        decode_and_downscale(b"this is not an image")


# --------------------------------------------------------------------------
# Outbound HTTP timeouts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", ["nvidia_vision", "plate_recognizer"])
def test_provider_requests_specify_a_timeout(module_name):
    """
    requests.post without timeout= waits forever. One stalled provider hangs the
    worker thread for the life of the process — the single most common cause of
    a service that is 'up' but answering nothing.
    """
    module = __import__(
        f"app.services.strategies.{module_name}", fromlist=[module_name]
    )
    source = inspect.getsource(module)
    assert "requests.post(" in source
    assert "timeout=" in source, (
        f"{module_name} calls requests.post without a timeout."
    )


def test_timeout_settings_are_bounded():
    assert 0 < settings.HTTP_CONNECT_TIMEOUT <= 10
    assert 0 < settings.HTTP_READ_TIMEOUT <= 30
