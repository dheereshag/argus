import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastanpr import FastANPR

app = FastAPI(
    title="Argus - Automatic Number Plate Recognition (ANPR) API",
    description="FastAPI application using FastANPR for vehicle license plate detection and text recognition.",
)

# Global lazy-loaded FastANPR instance
_anpr_instance = None


def get_anpr() -> FastANPR:
    global _anpr_instance
    if _anpr_instance is None:
        _anpr_instance = FastANPR(device="cpu")
    return _anpr_instance


class RecogniseRequest(BaseModel):
    image: str  # Base64 encoded image string


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Argus FastANPR License Plate Recognition API",
        "endpoints": {
            "recognise": "POST /recognise - JSON payload with base64 encoded image {'image': '...'}",
            "detect": "POST /detect - Multipart file upload",
            "docs": "/docs",
        },
    }


@app.post("/recognise")
async def recognise_base64_image(payload: RecogniseRequest):
    """
    Accepts a JSON payload containing a base64 encoded image:
    `{"image": "<base64_encoded_string>"}`

    Returns detected license plates matching the exact schema:
    ```json
    {
        "number_plates": [
            {
                "det_box": [682, 414, 779, 455],
                "det_conf": 0.29964497685432434,
                "rec_poly": [[688, 420], [775, 420], [775, 451], [688, 451]],
                "rec_text": "BVH826",
                "rec_conf": 0.940690815448761
            }
        ]
    }
    ```
    """
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(payload.image)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run FastANPR inference
        anpr = get_anpr()
        results = await anpr.run(image_bgr)

        number_plates = []
        if results and len(results) > 0:
            for plate in results[0]:
                number_plates.append({
                    "det_box": plate.det_box,
                    "det_conf": plate.det_conf,
                    "rec_poly": plate.rec_poly,
                    "rec_text": plate.rec_text,
                    "rec_conf": plate.rec_conf,
                })

        return {"number_plates": number_plates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect")
async def detect_number_plate_file(file: UploadFile = File(...)):
    """
    Upload an image file directly (multipart/form-data) to run ANPR.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        anpr = get_anpr()
        results = await anpr.run(image_bgr)

        number_plates = []
        if results and len(results) > 0:
            for plate in results[0]:
                number_plates.append({
                    "det_box": plate.det_box,
                    "det_conf": plate.det_conf,
                    "rec_poly": plate.rec_poly,
                    "rec_text": plate.rec_text,
                    "rec_conf": plate.rec_conf,
                })

        return {
            "filename": file.filename,
            "plates_detected_count": len(number_plates),
            "number_plates": number_plates,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
