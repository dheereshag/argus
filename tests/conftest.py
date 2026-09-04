import io

import pytest
from PIL import Image


def create_test_jpeg(width: int, height: int, color: tuple[int, int, int] = (100, 100, 100)) -> bytes:
    """Generates valid JPEG image bytes of specified dimensions for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes():
    """Generates a small valid RGB JPEG image in memory for testing."""
    return create_test_jpeg(100, 100, color=(255, 0, 0))


@pytest.fixture
def sample_png_bytes():
    """Generates a small valid PNG image in memory for testing."""
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    """Provides a TestClient instance for testing FastAPI endpoints."""
    from fastapi.testclient import TestClient

    from app.server import app

    with TestClient(app) as test_client:
        yield test_client
