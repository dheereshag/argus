"""Tests for cross-class vehicle spatial deduplication and duplicate plate suppression."""

from app.services.detector import VehicleDetector
from app.services.detector.geometry import box_iou
from app.services.detector.parser import _dedup_vehicles


def test_box_iou_identical_and_disjoint():
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    box3 = (200, 200, 300, 300)

    assert box_iou(box1, box2) == 1.0
    assert box_iou(box1, box3) == 0.0


def test_box_iou_partial_overlap():
    box1 = (0, 0, 100, 100)
    box2 = (50, 0, 150, 100)
    # Intersection: 50 * 100 = 5000, Union: 10000 + 10000 - 5000 = 15000 -> 1/3
    assert abs(box_iou(box1, box2) - (1.0 / 3.0)) < 1e-4


def test_dedup_overlapping_bus_and_truck():
    # Truck with high confidence and bus duplicate with lower confidence
    truck = (0.76, 100000, "truck", (10, 50, 390, 450))
    bus = (0.35, 99000, "bus", (8, 52, 388, 448))

    deduped = _dedup_vehicles([truck, bus])
    assert len(deduped) == 1
    assert deduped[0][0] == "truck"
    assert deduped[0][1] == (10, 50, 390, 450)


def test_detect_image_4_trucks_without_bus():
    detector = VehicleDetector()
    result = detector.detect("tests/4.jpg")

    # Image 4 has two trucks side by side on scale; bus duplicates must be suppressed
    assert len(result.vehicles) == 2
    types = [v.vehicle_type for v in result.vehicles]
    assert types == ["truck", "truck"]
    assert "bus" not in types
