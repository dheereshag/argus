"""Spatial association of full-frame OCR plates to detected vehicle bounding boxes."""

from app.core.config import settings
from app.schemas import DetectedVehicle, PlateResult
from app.services.detector.geometry import is_contained
from app.services.pipeline.helpers import validate_plate_results


def _bumper_match(b: tuple, v: DetectedVehicle) -> tuple[int, int] | None:
    vx1, vy1, vx2, vy2 = v.box
    ix = max(0, min(b[2], vx2) - max(b[0], vx1))
    if (ix / max(1, b[2] - b[0])) >= 0.50 and b[1] >= vy1 and (dy := max(0, b[1] - vy2)) <= 0.80 * max(1, vy2 - vy1):
        return (dy, -ix)
    return None


def _owner_vehicle(box: tuple | None, vehicles: list[DetectedVehicle]) -> DetectedVehicle | None:
    if not box:
        return vehicles[0] if len(vehicles) == 1 else None
    direct = next((v for v in vehicles if is_contained(box, v.box, 0.50)), None)
    if direct is not None:
        return direct
    cands = [(m, v) for v in vehicles if (m := _bumper_match(box, v)) is not None]
    return min(cands, key=lambda c: c[0])[1] if cands else None


def _dedup(results: list[PlateResult]) -> list[PlateResult]:
    seen: set[str] = set()
    out: list[PlateResult] = []
    for r in results:
        if r.plate is None or r.plate not in seen:
            if r.plate:
                seen.add(r.plate)
            out.append(r)
    return out


def associate_fullframe_plates(
    full_frame_raw: list[dict],
    vehicles: list[DetectedVehicle],
    crop_results: list[PlateResult],
    plated: set[int] | None = None,
) -> list[PlateResult]:
    """Merge full-frame OCR results with per-vehicle crop results via spatial association."""
    plated_ids = set(plated) if plated is not None else {id(v) for r in crop_results if r.plate for v in vehicles if r.vehicle_type == v.vehicle_type}
    extra: list[PlateResult] = []
    for raw in full_frame_raw:
        veh = _owner_vehicle(raw.get("box"), vehicles)
        if res := validate_plate_results([raw], veh.vehicle_type if veh else None):
            if veh:
                plated_ids.add(id(veh))
            extra.extend(res)

    combined = _dedup(list(crop_results) + extra)
    if settings.INCLUDE_UNIDENTIFIED_VEHICLES:
        combined.extend(PlateResult(plate=None, vehicle_type=v.vehicle_type) for v in vehicles if id(v) not in plated_ids)
    return combined
