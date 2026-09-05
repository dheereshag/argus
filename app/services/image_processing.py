"""
Image processing utilities and security validation for Argus ANPR.

Provides functions for:
  - Header inspection (probing dimensions and formats without full pixel rasterization).
  - Upload payload safety enforcement (decompression bomb protection, file size caps).
  - High-quality Lanczos downscaling to bound inference latency.
  - Multi-source RGB normalization supporting bytes, file paths, PIL Images, and OpenCV numpy arrays.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.constants import ALLOWED_IMAGE_FORMATS, ALLOWED_IMAGE_MIME_TYPES
from app.core.config import settings
from app.core.contracts import ensure, require
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

# Polymorphic input type accepted by image loading functions
type ImageInput = str | bytes | Image.Image | np.ndarray

# Enforce strict pixel ceiling in Pillow to guard against malicious decompression bombs
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

# Allow Pillow to read truncated/partially recovered image files rather than failing outright
object.__setattr__(ImageFile, "LOAD_TRUNCATED_IMAGES", True)


def probe_image(image_bytes: bytes) -> tuple[str, int, int]:
    """
    Inspect image metadata from binary stream without decoding the full pixel raster.

    Reads file headers to verify format, width, and height with minimal CPU and memory overhead.

    Args:
        image_bytes: Raw binary bytes of the image file.

    Returns:
        tuple[str, int, int]: (format_name, pixel_width, pixel_height).

    Raises:
        InvalidImageError: If the byte stream is empty, corrupted, or unsupported format.
        PayloadTooLargeError: If bytes exceed MAX_UPLOAD_BYTES or dimensions trigger DecompressionBombError.
    """
    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    # Reject oversized payloads before attempting header parse
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


def validate_image_upload(image_bytes: bytes, content_type: str | None = None) -> str:
    """
    Validate that uploaded file bytes constitute an allowed image format and MIME type.

    Args:
        image_bytes: Raw binary bytes from the upload stream.
        content_type: Optional HTTP Content-Type header string (e.g. 'image/jpeg; charset=utf-8').

    Returns:
        str: Uppercase image format (e.g., 'JPEG', 'PNG').

    Raises:
        InvalidImageError: If MIME type or image format fails validation.
    """
    if content_type:
        # Strip trailing parameters such as charset
        clean_type = content_type.split(";")[0].strip().lower()
        if clean_type != "application/octet-stream" and clean_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise InvalidImageError(
                f"Unsupported content type '{content_type}'. Allowed types: image/jpeg, image/png, image/webp, image/bmp."
            )

    fmt, _, _ = probe_image(image_bytes)
    return fmt


def decode_and_downscale(image_bytes: bytes, max_edge: int | None = None) -> bytes:
    """
    Validate an uploaded image and return downscaled JPEG bytes bounded by max_edge.

    Maintains original aspect ratio using high-quality Lanczos resampling.
    Ensures input frames sent to YOLO and OCR engines are bounded in memory and compute footprint.

    Args:
        image_bytes: Raw image file bytes.
        max_edge: Maximum allowed length for the longest edge in pixels (defaults to settings.MAX_IMAGE_EDGE_PX).

    Returns:
        bytes: Encoded JPEG bytes of the bounded image.
    """
    max_edge = max_edge or settings.MAX_IMAGE_EDGE_PX
    require(max_edge > 0, f"max_edge must be positive, got {max_edge}")

    # Verify header bounds and total pixel budget before raster decompression
    _, width, height = probe_image(image_bytes)
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise PayloadTooLargeError(
            f"Image is {width}x{height} ({width * height} pixels); limit is {settings.MAX_IMAGE_PIXELS} pixels."
        )

    pil_img = load_rgb(image_bytes)
    # Scale down if either dimension exceeds max_edge
    if max(pil_img.size) > max_edge:
        pil_img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    ensure(min(pil_img.size) > 0, "downscaled image collapsed to zero size")
    ensure(max(pil_img.size) <= max_edge, f"downscale failed to bound edge to {max_edge}")
    return _to_jpeg_bytes(pil_img)


def load_rgb(image_input: ImageInput) -> Image.Image:
    """
    Decode polymorphic image input into an oriented PIL RGB image.

    Handles:
      - PIL Image: Converts to RGB.
      - NumPy ndarray: Converts BGR (OpenCV standard) or grayscale to RGB.
      - Raw bytes: In-memory stream decode with EXIF transposition.
      - File path (str): File handle read with EXIF transposition.

    Args:
        image_input: Image representation (str filepath, raw bytes, PIL Image, or NumPy array).

    Returns:
        Image.Image: Standardized 8-bit RGB Pillow Image.
    """
    # 1. PIL Image passthrough
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    # 2. NumPy ndarray conversion (typically OpenCV BGR or single-channel grayscale)
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
        return Image.fromarray(
            cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        )

    # 3. File path or raw bytes decoding with EXIF orientation correction
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
    """Encode a PIL Image into compressed JPEG bytes."""
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
