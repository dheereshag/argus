"""FastAPI middleware registration for CORS and request execution timing."""

import time
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core import constants


def register_middleware(app: FastAPI) -> None:
    """Register CORS and request timing middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=constants.CORS_ORIGINS,
        allow_credentials=constants.CORS_ALLOW_CREDENTIALS,
        allow_methods=constants.CORS_ALLOW_METHODS,
        allow_headers=constants.CORS_ALLOW_HEADERS,
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Any) -> Response:
        """Measure total HTTP request roundtrip time and set X-Process-Time-Ms header."""
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
        return response
