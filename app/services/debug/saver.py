"""Debug utility to save intermediate vehicle crops and preprocessed images to disk."""

import re
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.core.logging import logger


def _sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem or "image"
    return re.sub(r"[^\w\-]", "_", stem)[:50]


def save_debug_crop(
    img: Image.Image,
    tag: str,
    filename: str = "image.jpg",
    vehicle_idx: int = 0,
    force: bool = False,
) -> Path | None:
    """Save intermediate vehicle crop or preprocessed image to DEBUG_CROPS_DIR if enabled."""
    if not (settings.DEBUG_SAVE_CROPS or force):
        return None
    try:
        out_dir = Path(settings.DEBUG_CROPS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")[:19]
        stem = _sanitize_stem(filename)
        out_file = out_dir / f"{ts}_{stem}_veh{vehicle_idx}_{tag}.jpg"
        to_save = img.convert("RGB") if img.mode != "RGB" else img
        to_save.save(out_file, format="JPEG", quality=95)
        logger.debug(f"Saved debug crop: {out_file}")
        return out_file
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as exc:
        logger.warning(f"Failed to save debug crop '{filename}': {exc}")
        return None
