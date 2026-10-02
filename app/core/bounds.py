"""Sequence bounds enforcement for defensive runtime bounding."""

from collections.abc import Sequence

from app.core.contracts import require
from app.core.logging import logger


def bounded[T](items: Sequence[T] | None, limit: int, what: str) -> Sequence[T]:
    """Enforce fixed upper bounds on item sequences at the data source."""
    require(limit > 0, f"{what} limit must be positive, got {limit}")

    if not items:
        return []
    if len(items) <= limit:
        return items

    logger.warning(
        f"[bounds] {what}: {len(items)} exceeds cap of {limit}; "
        f"processing first {limit} and discarding {len(items) - limit}."
    )
    return items[:limit]
