"""Response construction and timing benchmark formatting."""

import time

from app.schemas import (
    DetectionResult,
    PlateResult,
    RecognitionResponse,
    RecognitionStatusEnum,
)


def _build_response(
    detection: DetectionResult,
    filename: str,
    start_time: float,
    *,
    success: bool,
    rejected: bool,
    status: RecognitionStatusEnum,
    results: list[PlateResult] | None = None,
) -> RecognitionResponse:
    """Assemble standard RecognitionResponse model with calculated latency."""
    return RecognitionResponse(
        success=success,
        rejected=rejected,
        status=status,
        human_count=detection.human_count,
        filename=filename,
        results=results or [],
        execution_time_ms=round((time.time() - start_time) * 1000, 2),
    )
