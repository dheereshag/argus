"""Response construction for thin ANPR results."""

import time

from app.schemas import PlateResult, RecognitionResponse


def _build_response(
    plates: list[str],
    start_time: float,
) -> RecognitionResponse:
    """Assemble thin RecognitionResponse containing plate results with execution latency."""
    elapsed = round((time.time() - start_time) * 1000, 2)
    results = [PlateResult(plate=p, execution_time_ms=elapsed) for p in plates]
    return RecognitionResponse(
        results=results,
        execution_time_ms=elapsed,
    )
