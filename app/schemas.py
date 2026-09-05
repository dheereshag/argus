"""
Domain models, internal dataclasses, and API response schemas for Argus ANPR.

This module defines:
  - Slotted dataclasses for internal pipeline stages (OCR tokens, candidate ranking, detection).
  - Pydantic models for REST API request validation and response serialisation.
  - Enumerations for pre-screening and recognition status outcomes.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True, frozen=True)
class OCRToken:
    """
    Slotted, immutable container for a single OCR text element.

    Attributes:
        text: Extracted raw text string.
        score: OCR model confidence score in range [0.0, 1.0].
        cx: Centroid X coordinate in pixel space, if available from bounding quad/box.
        cy: Centroid Y coordinate in pixel space, if available from bounding quad/box.
    """

    text: str
    score: float
    cx: float | None = None
    cy: float | None = None


@dataclass(slots=True)
class PlateCandidate:
    """
    Slotted candidate plate match paired with spatial position and ranking priority.

    Used when sorting competing plate interpretations.
    Lower rank index indicates higher precedence in regex normalization heuristics.

    Attributes:
        y_pos: Vertical position (centroid Y) in the image frame for top-to-bottom spatial ordering.
        rank: Priority rank from normalization heuristics (0 = exact/primary match).
        info: Parsed plate metadata dictionary containing 'plate', 'state', and 'raw_text'.
    """

    y_pos: float
    rank: int
    info: dict[str, Any]


@dataclass(slots=True)
class DetectionResult:
    """
    Stage 1 Result: YOLO v11 vehicle detection, occupancy verification, and vehicle cropping.

    Attributes:
        is_eligible: True if the frame passes all pre-screening policies and should proceed to OCR.
        status: Specific rejection or success status code enum.
        status_message: Descriptive explanation of the detection and occupancy evaluation.
        vehicle_detected: Whether at least one 4-wheeler vehicle (car, bus, truck) was localized.
        vehicle_type: Name of the primary vehicle category ('car', 'bus', 'truck') or None.
        human_detected: Whether any person was detected exceeding the confidence threshold.
        vehicle_count: Total number of valid 4-wheeler detections meeting the confidence threshold.
        vehicle_box: Clamped (x1, y1, x2, y2) bounding box of the primary vehicle crop.
        crop: Cropped PIL RGB Image containing only the primary vehicle area, or None.
    """

    is_eligible: bool
    status: RecognitionStatusEnum | None
    status_message: str
    vehicle_detected: bool
    vehicle_type: str | None
    human_detected: bool
    vehicle_count: int
    vehicle_box: tuple[int, int, int, int] | None = None
    crop: Any = None


class RecognitionStatusEnum(str, Enum):
    """Enumeration of possible pre-screening policy evaluations and recognition outcomes."""

    SUCCESS = "success"
    REJECTED_NO_FOUR_WHEELER = "rejected_no_four_wheeler"
    REJECTED_HUMAN_DETECTED = "rejected_human_detected"
    REJECTED_MULTIPLE_VEHICLES = "rejected_multiple_vehicles"
    NO_PLATE_DETECTED = "no_plate_detected"


class PlateResult(BaseModel):
    """Schema representing an extracted and verified Indian license plate."""

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
    """
    Top-level API response schema for license plate recognition requests.

    Provides end-to-end details of both Stage 1 (YOLO detection) and Stage 2 (OCR recognition).
    """

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
    """Standardized error payload returned across all HTTP exception handlers."""

    success: bool = Field(False, description="Always False for error responses")
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Exception class or category")
    details: Any = Field(None, description="Detailed validation or contextual error info")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the error",
    )
