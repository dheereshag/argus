from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True, frozen=True)
class OCRToken:
    """Slotted container for extracted OCR tokens."""

    text: str
    score: float
    cx: float | None = None
    cy: float | None = None


@dataclass(slots=True)
class PlateCandidate:
    """Slotted candidate plate match with vertical spatial priority ranking."""

    y_pos: float
    rank: int
    info: dict[str, Any]



class RecognitionStatusEnum(str, Enum):
    SUCCESS = "success"
    REJECTED_NO_FOUR_WHEELER = "rejected_no_four_wheeler"
    REJECTED_HUMAN_DETECTED = "rejected_human_detected"
    REJECTED_MULTIPLE_VEHICLES = "rejected_multiple_vehicles"
    NO_PLATE_DETECTED = "no_plate_detected"


class PlateResult(BaseModel):
    plate: str = Field(
        ..., description="Normalized Indian vehicle registration number (e.g., RJ09GA0165)", examples=["RJ09GA0165"]
    )
    state: str | None = Field(
        None, description="State or Union Territory full name (e.g., Rajasthan)", examples=["Rajasthan"]
    )
    raw_text: str | None = Field(
        None, description="Raw OCR text extracted from the image frame/crop", examples=["BP1-A2453"]
    )


class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="Status of the recognition request")
    rejected: bool = Field(
        False,
        description="Whether the image was rejected during pre-screening (e.g. human detected, no vehicle, multiple vehicles)",
    )
    status: RecognitionStatusEnum = Field(
        ..., description="Detailed status enum for pre-screening and recognition outcome"
    )
    status_message: str = Field(..., description="Human readable description of the status outcome")
    vehicle_detected: bool = Field(..., description="Whether a 4-wheeler vehicle was detected in the frame")
    vehicle_type: str | None = Field(
        None, description="Specific type of 4-wheeler vehicle detected (e.g., 'car', 'bus', 'truck')", examples=["car"]
    )
    human_detected: bool = Field(..., description="Whether a human presence was detected in the frame")
    filename: str = Field(..., description="Name of the processed image file")
    results: list[PlateResult] = Field(default_factory=list, description="Extracted license plate details")
    execution_time_ms: float | None = Field(None, description="Processing duration in milliseconds")


class APIErrorResponse(BaseModel):
    success: bool = Field(False, description="Always False for error responses")
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Exception class or category")
    details: Any = Field(None, description="Detailed validation or contextual error info")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the error",
    )
