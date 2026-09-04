import io

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

# Python 3.12 Type Aliases
type BoundingBox = tuple[int, int, int, int]
type ImageInput = str | bytes | Image.Image | np.ndarray

# Decompression-bomb guard. Pillow's default limit only emits a warning;
# this makes an oversized image raise before allocation.
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
object.__setattr__(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

# A crop narrower than this cannot contain a readable plate. Used to reject
# degenerate boxes rather than feeding a 2-pixel sliver to downstream models.
MIN_CROP_EDGE_PX = 8


def decode_and_downscale(
    image_bytes: bytes,
    max_edge: int | None = None,
) -> bytes:
    """
    Validate an uploaded image and return normalised JPEG bytes.
    Guards decode against decompression bombs, applies EXIF orientation, and
    downscales so the longest edge is at most `max_edge`.
    """
    max_edge = max_edge or settings.MAX_IMAGE_EDGE_PX
    require(max_edge > 0, f"max_edge must be positive, got {max_edge}")
    require(bool(image_bytes), "decode_and_downscale received empty bytes")

    # Header probe without full pixel buffer allocation
    try:
        with io.BytesIO(image_bytes) as buf, Image.open(buf) as probe:
            width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}") from exc

    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )

    try:
        pil_img = load_rgb(image_bytes)
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}") from exc

    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return _to_jpeg_bytes(pil_img)


def clamp_box(
    box: BoundingBox | None,
    width: int,
    height: int,
) -> BoundingBox | None:
    """
    Clamp an xyxy box to real image bounds.
    Returns None when the box is malformed or degenerate after clamping.
    """
    require(width > 0 and height > 0, f"image dimensions must be positive, got {width}x{height}")

    if not box or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None

    # Handle corner-swapped coordinates
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 - x1 < MIN_CROP_EDGE_PX or y2 - y1 < MIN_CROP_EDGE_PX:
        return None

    return (x1, y1, x2, y2)


def load_rgb(image_input: ImageInput) -> Image.Image:
    """
    Decode to an oriented RGB image, releasing the source handle immediately.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
        return Image.fromarray(
            cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        )

    if isinstance(image_input, bytes):
        with io.BytesIO(image_input) as buf, Image.open(buf) as img:
            img.load()
            oriented = ImageOps.exif_transpose(img)
            return oriented.convert("RGB")

    with open(image_input, "rb") as fh, Image.open(fh) as img:
        img.load()
        oriented = ImageOps.exif_transpose(img)
        return oriented.convert("RGB")


def _to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
