"""Polymorphic image decoding into oriented RGB Pillow images."""

import io

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.core.constants import MAX_IMAGE_PIXELS
from app.core.exceptions import InvalidImageError, PayloadTooLargeError

type ImageInput = str | bytes | Image.Image | np.ndarray

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
object.__setattr__(ImageFile, "LOAD_TRUNCATED_IMAGES", True)


def _from_ndarray(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.shape[2] == 3 else arr)


def load_rgb(image_input: ImageInput) -> Image.Image:
    """Decode polymorphic image input into an oriented PIL RGB image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        return _from_ndarray(image_input)

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
