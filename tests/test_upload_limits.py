"""
Tests for payload budgets, pixel caps, and image decode validation.
"""

import pytest

from app.core.constants import MAX_UPLOAD_BYTES
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.services.image_processing import decode_image
from app.services.pipeline import recognize_plate_image
from tests.conftest import create_test_jpeg as _jpeg

# --------------------------------------------------------------------------
# Upload size & limits
# --------------------------------------------------------------------------


def test_oversized_upload_is_rejected():
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_UPLOAD_BYTES + 1024)
    with pytest.raises(PayloadTooLargeError):
        recognize_plate_image(oversized, filename="huge.jpg")


def test_empty_upload_is_rejected():
    with pytest.raises(InvalidImageError):
        recognize_plate_image(b"", filename="empty.jpg")


# --------------------------------------------------------------------------
# Decode bombs and pixel cap
# --------------------------------------------------------------------------


def test_pixel_budget_is_enforced(monkeypatch):
    """
    A small file can declare enormous dimensions. Guard on the pixel count from
    the header, before the full decode allocates anything.
    """
    monkeypatch.setattr("app.services.image_processing.transformer.MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(PayloadTooLargeError):
        decode_image(_jpeg(200, 200))


def test_full_resolution_image_is_not_modified():
    """1920x1080 (camera native resolution) must pass through without any modification."""
    from PIL import Image

    out = decode_image(_jpeg(1920, 1080))
    assert isinstance(out, Image.Image)
    assert out.size == (1920, 1080)


def test_small_image_is_not_upscaled():
    from PIL import Image

    out = decode_image(_jpeg(320, 240))
    assert isinstance(out, Image.Image)
    assert out.size == (320, 240)


def test_undecodable_bytes_raise_invalid_image():
    with pytest.raises(InvalidImageError):
        decode_image(b"this is not an image")
