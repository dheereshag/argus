from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.schemas.plate import RecognitionResponse
from app.services.image_processing import validate_image_upload
from app.services.pipeline import recognize_plate_image

router = APIRouter()


@router.get(
    "/",
    summary="Service Information",
    description="Root metadata information for the Argus ANPR API.",
    tags=["Info"],
)
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
    }


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
    image_bytes = await file.read()
    validate_image_upload(image_bytes, content_type=file.content_type)
    filename = file.filename or "uploaded_image.jpg"

    return recognize_plate_image(
        image_input=image_bytes,
        filename=filename,
    )
