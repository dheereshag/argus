import os
from unittest.mock import patch

from app.schemas import RecognitionResponse


def test_recognize_empty_file(client):
    response = client.post(
        "/recognize",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "empty" in data["message"].lower()


def test_recognize_invalid_image_bytes(client):
    response = client.post(
        "/recognize",
        files={"file": ("corrupt.jpg", b"not-a-valid-image-stream", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False


def test_recognize_payload_too_large(client):
    # Simulate a file larger than MAX_UPLOAD_BYTES
    with patch("app.services.image_processing.security.MAX_UPLOAD_BYTES", 100):
        large_bytes = b"X" * 200
        response = client.post(
            "/recognize",
            files={"file": ("large.jpg", large_bytes, "image/jpeg")},
        )
        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert "exceeds" in data["message"].lower()


def test_recognize_sample_image(client, sample_image_bytes):
    response = client.post(
        "/recognize",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = RecognitionResponse.model_validate(data)
    assert validated.filename == "test.jpg"
    # Plain red box has no vehicle and no plate
    assert validated.results == []
    assert validated.humans_outside == 0
    assert validated.humans_inside == 0


def test_recognize_real_image_if_present(client):
    image_path = os.path.join("tests", "1.jpg")
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/recognize",
        files={"file": ("1.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = RecognitionResponse.model_validate(data)
    assert validated.filename == "1.jpg"
    assert isinstance(validated.humans_outside, int)
    assert isinstance(validated.humans_inside, int)
    assert isinstance(validated.results, list)
    assert validated.execution_time_ms is not None


def test_recognize_rejects_unsupported_content_type(client):
    response = client.post(
        "/recognize",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "unsupported content type" in data["message"].lower()


def test_recognize_rejects_spoofed_mime_type(client):
    # Spoof content type as image/jpeg but send text bytes
    response = client.post(
        "/recognize",
        files={"file": ("fake.jpg", b"not-a-real-jpeg", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False


def test_recognize_rejects_unsupported_image_format(client):
    import io

    from PIL import Image

    img = Image.new("RGB", (50, 50), color=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    gif_bytes = buf.getvalue()

    response = client.post(
        "/recognize",
        files={"file": ("animated.gif", gif_bytes, "image/gif")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "unsupported" in data["message"].lower()


def test_recognize_accepts_valid_png(client, sample_png_bytes):
    response = client.post(
        "/recognize",
        files={"file": ("test.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = RecognitionResponse.model_validate(data)
    assert validated.filename == "test.png"
