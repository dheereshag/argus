"""Spatial Non-Maximum Suppression (NMS) for license plate candidates."""

from app.schemas import PlateCandidate
from app.services.detector.geometry import box_iou, is_contained


def _is_candidate_overlapping(cand: PlateCandidate, sel: PlateCandidate) -> bool:
    """Check if candidate spatially overlaps or duplicates an already selected plate."""
    cand_plate = cand.info.get("plate")
    sel_plate = sel.info.get("plate")
    if cand_plate and sel_plate and cand_plate == sel_plate:
        return True

    cb, sb = cand.box, sel.box
    return bool(cb is not None and sb is not None and (box_iou(cb, sb) > 0.15 or is_contained(cb, sb, 0.60) or is_contained(sb, cb, 0.60)))


def suppress_overlapping_candidates(candidates: list[PlateCandidate]) -> list[PlateCandidate]:
    """Select non-overlapping plate candidates using greedy Spatial NMS."""
    selected: list[PlateCandidate] = []
    for cand in candidates:
        if not any(_is_candidate_overlapping(cand, sel) for sel in selected):
            selected.append(cand)
    return selected
