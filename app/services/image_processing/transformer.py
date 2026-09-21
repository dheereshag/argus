"""Image downscaling and JPEG serialization."""

import io

from PIL import Image

from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import PayloadTooLargeError
from app.services.image_processing.loader import load_rgb
from app.services.image_processing.security import probe_image


def _to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Encode a PIL Image into compressed JPEG bytes."""
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def decode_and_downscale(image_bytes: bytes, max_edge: int | None = None) -> bytes:
    """Validate an uploaded image and return downscaled JPEG bytes bounded by max_edge."""
    max_edge = max_edge or settings.MAX_IMAGE_EDGE_PX
    require(max_edge > 0, f"max_edge must be positive, got {max_edge}")

    _, width, height = probe_image(image_bytes)
    if width * height > settings.MAX_IMAGE_PIXELS:
        limit = settings.MAX_IMAGE_PIXELS
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {limit} pixels."
        )

    pil_img = load_rgb(image_bytes)
    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return _to_jpeg_bytes(pil_img)
