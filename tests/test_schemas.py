from app.schemas import (
    DetectionResult,
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)


def test_recognition_status_enum():
    assert RecognitionStatusEnum.SUCCESS.value == "success"
    assert RecognitionStatusEnum.REJECTED_HUMAN_DETECTED.value == "rejected_human_detected"
    assert RecognitionStatusEnum.REJECTED_NO_FOUR_WHEELER.value == "rejected_no_four_wheeler"
    assert RecognitionStatusEnum.REJECTED_MULTIPLE_VEHICLES.value == "rejected_multiple_vehicles"
    assert RecognitionStatusEnum.NO_PLATE_DETECTED.value == "no_plate_detected"


def test_plate_result_valid():
    res = PlateResult(plate="RJ09GA0165", state="Rajasthan")
    assert res.plate == "RJ09GA0165"
    assert res.state == "Rajasthan"


def test_recognition_response_valid():
    resp = RecognitionResponse(
        success=True,
        status=RecognitionStatusEnum.SUCCESS,
        vehicle_type="car",
        vehicle_count=1,
        human_count=0,
        filename="test.jpg",
        results=[PlateResult(plate="RJ09GA0165", state="Rajasthan")],
        execution_time_ms=123.45,
    )
    assert resp.success is True
    assert resp.status == RecognitionStatusEnum.SUCCESS
    assert resp.vehicle_count == 1
    assert resp.human_count == 0
    assert len(resp.results) == 1
    assert resp.results[0].plate == "RJ09GA0165"


def test_detection_result_valid():
    det = DetectionResult(
        is_eligible=True,
        status=None,
        vehicle_type="car",
        vehicle_count=1,
        human_count=0,
    )
    assert det.is_eligible is True
    assert det.status is None
    assert det.vehicle_count == 1
    assert det.human_count == 0
