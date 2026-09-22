"""Human spatial occupancy classification (inside cabin vs outside vehicle)."""

from app.services.detector.geometry import BoundingBox, is_contained


def partition_humans(
    human_candidates: list[BoundingBox],
    vehicles: list[tuple[str, BoundingBox]],
) -> tuple[int, int]:
    """Classify humans into (outside_count, inside_count) based on cabin containment."""
    if not human_candidates:
        return 0, 0
    if not vehicles:
        return len(human_candidates), 0

    inside = sum(1 for hb in human_candidates if any(is_contained(hb, vb) for _, vb in vehicles))
    return len(human_candidates) - inside, inside

