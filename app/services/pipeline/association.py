"""Spatial association of full-frame OCR plates to detected vehicle bounding boxes."""

from pydantic import ValidationError

from app.schemas import DetectedVehicle, PlateResult
from app.services.detector.geometry import is_contained


def _to_plate_result(raw: dict, vehicle_type: str | None) -> PlateResult | None:
    plate = raw.get("plate")
    if not plate or plate == "N/A":
        return None
    entry = {**raw, **({"vehicle_type": vehicle_type} if vehicle_type and "vehicle_type" not in raw else {})}
    try:
        return PlateResult.model_validate(entry)
    except ValidationError:
        return None


def _owner_vehicle(box: tuple | None, vehicles: list[DetectedVehicle]) -> DetectedVehicle | None:
    if not box:
        return vehicles[0] if len(vehicles) == 1 else None
    return next((v for v in vehicles if is_contained(box, v.box, 0.50)), None)


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
    plated_ids = set(plated) if plated is not None else {
        id(v) for r in crop_results if r.plate for v in vehicles if r.vehicle_type == v.vehicle_type
    }
    extra: list[PlateResult] = []
    for raw in full_frame_raw:
        veh = _owner_vehicle(raw.get("box"), vehicles)
        result = _to_plate_result(raw, veh.vehicle_type if veh else None)
        if result is None:
            continue
        if veh:
            plated_ids.add(id(veh))
        extra.append(result)

    combined = _dedup(list(crop_results) + extra)
    combined.extend(PlateResult(plate=None, vehicle_type=v.vehicle_type) for v in vehicles if id(v) not in plated_ids)
    return combined
