"""In-memory image downscaling and boundary validation."""

from PIL import Image

from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import PayloadTooLargeError
from app.services.image_processing.loader import load_rgb
from app.services.image_processing.security import probe_image


def decode_and_downscale(image_bytes: bytes, max_edge: int | None = None) -> Image.Image:
    """Validate an uploaded image and return downscaled in-memory PIL Image bounded by max_edge."""
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
        resample = getattr(Image.Resampling, settings.IMAGE_RESAMPLE_FILTER, Image.Resampling.BILINEAR)
        pil_img.thumbnail((max_edge, max_edge), resample)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return pil_img

