from app.schemas import (
    DetectedVehicle,
    DetectionResult,
    PlateResult,
    RecognitionResponse,
)


def test_plate_result_valid():
    res = PlateResult(plate="RJ09GA0165", execution_time_ms=31.42)
    assert res.plate == "RJ09GA0165"
    assert res.execution_time_ms == 31.42


def test_recognition_response_valid():
    resp = RecognitionResponse(
        results=[PlateResult(plate="RJ09GA0165", execution_time_ms=123.45)],
        execution_time_ms=123.45,
    )
    assert resp.execution_time_ms == 123.45
    assert len(resp.results) == 1
    assert resp.results[0].plate == "RJ09GA0165"
    assert resp.results[0].execution_time_ms == 123.45


def test_detection_result_valid():
    det = DetectionResult(
        vehicles=[DetectedVehicle(vehicle_type="car", box=(10, 10, 50, 50))],
    )
    assert len(det.vehicles) == 1
    assert det.vehicles[0].vehicle_type == "car"
    assert det.vehicles[0].box == (10, 10, 50, 50)
