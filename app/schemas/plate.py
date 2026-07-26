from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class ProviderEnum(str, Enum):
    PLATERECOGNIZER = "platerecognizer"
    NVIDIA = "nvidia"

class PlateResult(BaseModel):
    plate: str = Field(..., description="Normalized Indian vehicle registration number (e.g., RJ09GA0165)", example="RJ09GA0165")
    state: Optional[str] = Field(None, description="State or Union Territory full name (e.g., Rajasthan)", example="Rajasthan")

class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="Status of the recognition request")
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
