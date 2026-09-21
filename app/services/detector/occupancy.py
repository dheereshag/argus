"""Weighbridge occupancy business rule evaluation."""

from app.core.config import settings
from app.core.logging import logger
from app.schemas import RecognitionStatusEnum
from app.services.detector.geometry import BoundingBox


def evaluate_occupancy(
    human_count: int,
    vehicles: list[tuple[str, BoundingBox]],
) -> tuple[bool, RecognitionStatusEnum | None]:
    """Evaluate weighbridge occupancy policies based on detection counts and thresholds."""
    vehicle_count = len(vehicles)
    human_limit = settings.MAX_ALLOWED_HUMANS
    vehicle_max = settings.MAX_ALLOWED_VEHICLES
    vehicle_min = settings.MIN_ALLOWED_VEHICLES

    if human_limit is not None and human_count > human_limit:
        logger.warning(f"Rejected frame: Human count ({human_count}) exceeded limit ({human_limit}).")
        return False, RecognitionStatusEnum.REJECTED_HUMAN_DETECTED

    if vehicle_max is not None and vehicle_count > vehicle_max:
        types_str = ", ".join(v[0] for v in vehicles)
        logger.warning(f"Rejected frame: {vehicle_count} vehicles detected ({types_str}).")
        return False, RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES

    if vehicle_count < vehicle_min:
        return False, RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER

    return True, None
