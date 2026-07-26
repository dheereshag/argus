from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class ProviderEnum(str, Enum):
    PLATERECOGNIZER = "platerecognizer"
    NVIDIA = "nvidia"

class RecognitionStatusEnum(str, Enum):
    SUCCESS = "success"
    REJECTED_NO_FOUR_WHEELER = "rejected_no_four_wheeler"
    REJECTED_HUMAN_DETECTED = "rejected_human_detected"
    NO_PLATE_DETECTED = "no_plate_detected"

class PlateResult(BaseModel):
    plate: str = Field(..., description="Normalized Indian vehicle registration number (e.g., RJ09GA0165)", example="RJ09GA0165")
    state: Optional[str] = Field(None, description="State or Union Territory full name (e.g., Rajasthan)", example="Rajasthan")

class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="Status of the recognition request")
    status: RecognitionStatusEnum = Field(..., description="Detailed status enum for pre-screening and recognition outcome")
    status_message: str = Field(..., description="Human readable description of the status outcome")
    vehicle_detected: bool = Field(..., description="Whether a 4-wheeler vehicle was detected in the frame")
    human_detected: bool = Field(..., description="Whether a human presence was detected in the frame")
    filename: str = Field(..., description="Name of the processed image file")
    provider: ProviderEnum = Field(..., description="AI recognition provider engine used")
    results: List[PlateResult] = Field(default_factory=list, description="Extracted license plate details")
    execution_time_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field("healthy", example="healthy")
    version: str = Field("1.0.0", example="1.0.0")

class ProvidersResponse(BaseModel):
    available_providers: List[ProviderEnum] = Field(..., description="List of supported recognition providers")
    default_provider: ProviderEnum = Field(..., description="System default recognition provider")
