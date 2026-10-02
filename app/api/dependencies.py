"""FastAPI dependencies for request handling and concurrency throttling."""

import asyncio
from typing import Annotated

from fastapi import Depends

from app.core import constants

_inference_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    """Return singleton asyncio semaphore limiting concurrent AI pipeline executions."""
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(constants.MAX_CONCURRENT_INFERENCES)
    return _inference_semaphore


type InferenceSemaphore = Annotated[asyncio.Semaphore, Depends(get_semaphore)]
