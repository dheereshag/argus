from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.schemas.plate import RecognitionResponse
from app.services.image_processing import validate_image_upload
from app.services.pipeline import recognize_plate_image

router = APIRouter()


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Recognize Vehicle License Plate",
    description=(
        "Upload a vehicle image (JPEG, PNG, WebP, BMP) to execute YOLO v11 pre-screening "
        "and automated license plate recognition."
    ),
    tags=["Recognition"],
)
async def recognize_plate(
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, BMP)")],
) -> RecognitionResponse:
    if file.size is not None and file.size > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise InvalidImageError("Uploaded image file is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"Image exceeds maximum permitted size of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    # Early image validation: verify MIME type and image header signatures before offloading to worker pool
    validate_image_upload(image_bytes, content_type=file.content_type)

    filename = file.filename or "uploaded_image.jpg"

    # Run heavy CV / OCR inference in worker threadpool to prevent blocking asyncio loop
    response: RecognitionResponse = await run_in_threadpool(
        recognize_plate_image,
        image_input=image_bytes,
        filename=filename,
    )

    return response
