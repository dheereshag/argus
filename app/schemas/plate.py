from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ProviderEnum(str, Enum):
    PLATERECOGNIZER = "platerecognizer"
    NVIDIA = "nvidia"
    PADDLEOCR = "paddleocr"

class RecognitionStatusEnum(str, Enum):
    SUCCESS = "success"
    REJECTED_NO_FOUR_WHEELER = "rejected_no_four_wheeler"
    REJECTED_HUMAN_DETECTED = "rejected_human_detected"
    REJECTED_MULTIPLE_VEHICLES = "rejected_multiple_vehicles"
    NO_PLATE_DETECTED = "no_plate_detected"

class PlateResult(BaseModel):
    plate: str = Field(..., description="Normalized Indian vehicle registration number (e.g., RJ09GA0165)", examples=["RJ09GA0165"])
    state: Optional[str] = Field(None, description="State or Union Territory full name (e.g., Rajasthan)", examples=["Rajasthan"])
    raw_text: Optional[str] = Field(None, description="Raw OCR text extracted from the image frame/crop", examples=["BP1-A2453"])

class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="Status of the recognition request")
    status: RecognitionStatusEnum = Field(..., description="Detailed status enum for pre-screening and recognition outcome")
    status_message: str = Field(..., description="Human readable description of the status outcome")
    vehicle_detected: bool = Field(..., description="Whether a 4-wheeler vehicle was detected in the frame")
    vehicle_type: Optional[str] = Field(None, description="Specific type of 4-wheeler vehicle detected (e.g., 'car', 'bus', 'truck')", examples=["car"])
    human_detected: bool = Field(..., description="Whether a human presence was detected in the frame")
    filename: str = Field(..., description="Name of the processed image file")
    provider: ProviderEnum = Field(..., description="AI recognition provider engine used")
    results: List[PlateResult] = Field(default_factory=list, description="Extracted license plate details")
    execution_time_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field("healthy", examples=["healthy"])
    version: str = Field("1.0.0", examples=["1.0.0"])

class ProvidersResponse(BaseModel):
    available_providers: List[ProviderEnum] = Field(..., description="List of supported recognition providers")
    default_provider: ProviderEnum = Field(..., description="System default recognition provider")
