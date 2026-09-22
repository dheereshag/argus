from app.schemas import (
    DetectedVehicle,
    DetectionResult,
    PlateResult,
    RecognitionResponse,
)


def test_plate_result_valid():
    res = PlateResult(plate="RJ09GA0165", vehicle_type="car", state="Rajasthan")
    assert res.plate == "RJ09GA0165"
    assert res.vehicle_type == "car"
    assert res.state == "Rajasthan"


def test_plate_result_unread_vehicle():
    res = PlateResult(plate=None, vehicle_type="truck")
    assert res.plate is None
    assert res.vehicle_type == "truck"


def test_recognition_response_valid():
    resp = RecognitionResponse(
        filename="test.jpg",
        humans_outside=1,
        humans_inside=0,
        results=[PlateResult(plate="RJ09GA0165", vehicle_type="car", state="Rajasthan")],
        execution_time_ms=123.45,
    )
    assert resp.filename == "test.jpg"
    assert resp.humans_outside == 1
    assert resp.humans_inside == 0
    assert len(resp.results) == 1
    assert resp.results[0].plate == "RJ09GA0165"
    assert resp.results[0].vehicle_type == "car"


def test_detection_result_valid():
    det = DetectionResult(
        vehicles=[DetectedVehicle(vehicle_type="car", box=(10, 10, 50, 50))],
        humans_outside=2,
        humans_inside=1,
    )
    assert det.humans_outside == 2
    assert det.humans_inside == 1
    assert len(det.vehicles) == 1
    assert det.vehicles[0].vehicle_type == "car"
    assert det.vehicles[0].box == (10, 10, 50, 50)
