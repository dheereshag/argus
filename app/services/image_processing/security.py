"""Image security checks, format probing, and upload validation."""

import io

from PIL import Image

from app.constants import ALLOWED_IMAGE_FORMATS, ALLOWED_IMAGE_MIME_TYPES
from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError


def probe_image(image_bytes: bytes) -> tuple[str, int, int]:
    """Inspect image metadata from binary stream without decoding full pixel raster."""
    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise PayloadTooLargeError(f"Image exceeds maximum permitted size of {limit_mb} MB.")

    try:
        with io.BytesIO(image_bytes) as buf, Image.open(buf) as probe:
            fmt = probe.format
            if not fmt or fmt.upper() not in ALLOWED_IMAGE_FORMATS:
                allowed = ", ".join(sorted(ALLOWED_IMAGE_FORMATS))
                raise InvalidImageError(
                    f"Unsupported image format '{fmt}'. Allowed formats: {allowed}."
                )
            return fmt.upper(), probe.width, probe.height
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"Image dimensions exceed permitted budget: {exc}") from exc
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError(f"Uploaded file is not a valid image: {exc}") from exc


def validate_image_upload(image_bytes: bytes, content_type: str | None = None) -> str:
    """Validate that uploaded file bytes constitute an allowed image format and MIME type."""
    if content_type:
        clean_type = content_type.split(";")[0].strip().lower()
        if clean_type != "application/octet-stream" and clean_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise InvalidImageError(
                f"Unsupported content type '{content_type}'. Allowed types: image/jpeg, image/png, image/webp, image/bmp."
            )

    fmt, _, _ = probe_image(image_bytes)
    return fmt
