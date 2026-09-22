"""Image validation, decoding, and decompression bomb guard."""

from PIL import Image

from app.core.config import settings
from app.core.contracts import ensure
from app.core.exceptions import PayloadTooLargeError
from app.services.image_processing.loader import load_rgb
from app.services.image_processing.security import probe_image


def decode_image(image_bytes: bytes) -> Image.Image:
    """Validate an uploaded image and decode it to a full-resolution RGB PIL Image."""
    _, width, height = probe_image(image_bytes)
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )
    pil_img = load_rgb(image_bytes)
    ensure(min(pil_img.size) > 0, "decoded image has zero-size dimension")
    return pil_img
