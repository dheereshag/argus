"""Response construction and timing benchmark formatting."""

import time

from app.schemas import DetectionResult, PlateResult, RecognitionResponse


def _build_response(
    detection: DetectionResult,
    filename: str,
    start_time: float,
    results: list[PlateResult] | None = None,
) -> RecognitionResponse:
    """Assemble factual RecognitionResponse model with calculated latency."""
    return RecognitionResponse(
        filename=filename,
        humans_outside=detection.humans_outside,
        humans_inside=detection.humans_inside,
        results=results or [],
        execution_time_ms=round((time.time() - start_time) * 1000, 2),
    )
