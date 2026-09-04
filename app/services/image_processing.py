import io

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

# Python 3.12 Type Aliases
type ImageInput = str | bytes | Image.Image | np.ndarray

# Decompression-bomb guard. Pillow's default limit only emits a warning;
# this makes an oversized image raise before allocation.
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
object.__setattr__(ImageFile, "LOAD_TRUNCATED_IMAGES", True)

# Permitted image formats and MIME types
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/pjpeg",
        "image/png",
        "image/x-png",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
    }
)


def probe_image(image_bytes: bytes) -> tuple[str, int, int]:
    """
    Inspect image header without allocating full pixel buffers.
    Returns (format, width, height).
    """
    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        with io.BytesIO(image_bytes) as buf, Image.open(buf) as probe:
            fmt = probe.format
            if not fmt or fmt.upper() not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError(
                    f"Unsupported image format '{fmt}'. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
                )
            return fmt.upper(), probe.width, probe.height
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Uploaded file is not a valid image: {exc}") from exc


def validate_image_upload(
    image_bytes: bytes,
    content_type: str | None = None,
) -> str:
    """
    Validate that uploaded file bytes constitute a supported image.

    Checks declared MIME type (if provided and non-generic) and inspects image
    header without allocating pixel buffers. Returns the detected image format (e.g., 'JPEG').
    """
    if content_type:
        clean_type = content_type.split(";")[0].strip().lower()
        if clean_type != "application/octet-stream" and clean_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise InvalidImageError(
                f"Unsupported content type '{content_type}'. Allowed types: image/jpeg, image/png, image/webp, image/bmp."
            )

    fmt, _, _ = probe_image(image_bytes)
    return fmt


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

    _, width, height = probe_image(image_bytes)

    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )

    pil_img = load_rgb(image_bytes)

    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return _to_jpeg_bytes(pil_img)


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

    try:
        if isinstance(image_input, bytes):
            with io.BytesIO(image_input) as fh, Image.open(fh) as img:
                img.load()
                return ImageOps.exif_transpose(img).convert("RGB")
        with open(image_input, "rb") as fh, Image.open(fh) as img:
            img.load()
            return ImageOps.exif_transpose(img).convert("RGB")
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except Exception as exc:
        raise InvalidImageError(f"Could not decode uploaded image: {exc}") from exc


def _to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
