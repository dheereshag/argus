"""Vehicle license plate recognition REST routes."""

from typing import Annotated

from asyncer import asyncify
from fastapi import APIRouter, File, UploadFile

from app.api.dependencies import InferenceSemaphore
from app.schemas import RecognitionResponse
from app.services.image_processing import validate_image_upload
from app.services.pipeline import recognize_plate_image

router = APIRouter(tags=["Recognition"])


@router.post("/recognize", summary="Recognize Vehicle License Plate")
async def recognize_plate(
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, BMP)")],
    semaphore: InferenceSemaphore,
) -> RecognitionResponse:
    """Process an uploaded vehicle image through the 4-Tier Cascaded ANPR Pipeline."""
    image_bytes = await file.read()
    validate_image_upload(image_bytes, content_type=file.content_type)
    async with semaphore:
        return await asyncify(recognize_plate_image)(
            image_bytes,
            filename=file.filename or "image.jpg",
        )
