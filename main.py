import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from anpr import PlateRecognizerFactory

app = FastAPI(
    title="Argus ANPR Microservice",
    description="Automatic Number Plate Recognition Microservice supporting multiple AI recognition strategies (Plate Recognizer, NVIDIA Vision LLM).",
    version="1.0.0"
)

class RecognizedPlateItem(BaseModel):
    plate: str = Field(..., description="Normalized Indian vehicle license plate number")
    state: str = Field(..., description="State or Union Territory name")

class RecognizeResponse(BaseModel):
    success: bool
    filename: str
    provider: str
    results: List[RecognizedPlateItem]

@app.get("/")
def root():
    return {
        "service": "Argus ANPR Microservice",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/providers")
def list_providers():
    return {
        "available_providers": ["platerecognizer", "nvidia"],
        "default_provider": "platerecognizer"
    }

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_plate(
    file: UploadFile = File(..., description="Vehicle image file (JPEG, PNG)"),
    provider: Optional[str] = Query(
        None, 
        description="Recognition model provider: 'platerecognizer' or 'nvidia'. Uses default from .env if omitted."
    )
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is not a valid image format."
        )

    try:
        image_bytes = await file.read()
        
        # Instantiate model strategy via Factory
        recognizer = PlateRecognizerFactory.get_recognizer(provider)
        provider_name = provider or "platerecognizer (default)"

        # Perform recognition
        results = recognizer.recognize(image_bytes, filename=file.filename)

        return RecognizeResponse(
            success=True,
            filename=file.filename,
            provider=provider_name,
            results=results
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during recognition: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
