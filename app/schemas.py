"""
Domain models, internal dataclasses, and API response schemas for Argus ANPR.

This module defines:
  - Slotted dataclasses for internal pipeline stages (OCR tokens, candidate ranking, detection).
  - Pydantic models for REST API request validation and response serialisation.
  - Enumerations for pre-screening and recognition status outcomes.
"""

from dataclasses import dataclass, field
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
        box: (x1, y1, x2, y2) bounding box in pixel space.
    """

    text: str
    score: float
    cx: float | None = None
    cy: float | None = None
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class PlateCandidate:
    """
    Scored candidate license plate generated during spatial pairing and OCR analysis.

    Attributes:
        y_pos: Vertical centroid in pixels (used for positional prioritization).
        rank: Rule-based score reflecting plate syntax validity (higher is better).
        info: Structured metadata dictionary parsed from plate string.
        confidence: Average OCR confidence score for candidate tokens [0.0 - 1.0].
        box: Bounding box (x1, y1, x2, y2) enclosing the candidate tokens.
    """

    y_pos: float
    rank: int
    info: dict[str, Any]
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectedVehicle:
    """
    Stage 1 detected vehicle entity.

    Attributes:
        vehicle_type: Category of 4-wheeler ('car', 'bus', 'truck').
        box: Clamped (x1, y1, x2, y2) bounding box in original image space.
        crop: Cropped PIL RGB Image containing only this vehicle area, or None.
        crop_box: Padded (x1, y1, x2, y2) bounding box used for crop, or None.
    """

    vehicle_type: str
    box: tuple[int, int, int, int]
    crop: Any = None
    crop_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectionResult:
    """
    Stage 1 Result: YOLO26 vehicle detection, occupancy verification, and vehicle cropping.

    Attributes:
        is_eligible: True if the frame passes all pre-screening policies and should proceed to OCR.
        status: Specific rejection or success status code enum.
        vehicles: List of all detected 4-wheeler vehicles meeting confidence and size thresholds.
        human_count: Total number of valid human detections meeting the confidence threshold.
    """

    is_eligible: bool
    status: RecognitionStatusEnum | None
    vehicles: list[DetectedVehicle] = field(default_factory=list)
    human_count: int = 0


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
        description="Normalized Indian vehicle registration number (e.g., RJ09GA0165)", examples=["RJ09GA0165"]
    )
    vehicle_type: str | None = Field(
        None,
        description="Specific type of 4-wheeler vehicle detected (e.g., 'car', 'bus', 'truck')",
        examples=["car"],
    )
    state: str | None = Field(
        None, description="State or Union Territory full name (e.g., Rajasthan)", examples=["Rajasthan"]
    )
    raw_text: str | None = Field(
        None, description="Raw OCR text extracted from the image frame/crop", examples=["BP1-A2453"]
    )
    confidence: float | None = Field(
        None, description="Average OCR confidence score for plate characters [0.0 - 1.0]", examples=[0.98]
    )
    box: tuple[int, int, int, int] | None = Field(
        None, description="Bounding box (x1, y1, x2, y2) of plate in pixel space", examples=[(100, 200, 300, 250)]
    )


class RecognitionResponse(BaseModel):
    """
    Top-level API response schema for license plate recognition requests.

    Provides end-to-end details of both Stage 1 (YOLO detection) and Stage 2 (OCR recognition).
    """

    success: bool = Field(description="Status of the recognition request")
    rejected: bool = Field(
        False,
        description="Whether the image was rejected during pre-screening",
    )
    status: RecognitionStatusEnum = Field(
        description="Detailed status enum for pre-screening and recognition outcome"
    )
    human_count: int = Field(0, description="Total number of humans detected in the frame")
    filename: str = Field(description="Name of the processed image file")
    results: list[PlateResult] = Field(default_factory=list, description="Extracted license plate details")
    execution_time_ms: float | None = Field(None, description="Processing duration in milliseconds")


class APIErrorResponse(BaseModel):
    """Standardized error payload returned across all HTTP exception handlers."""

    success: bool = Field(False, description="Always False for error responses")
    status_code: int = Field(description="HTTP status code")
    message: str = Field(description="Human-readable error description")
    error_type: str = Field(description="Exception class or category")
    details: Any = Field(None, description="Detailed validation or contextual error info")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the error",
    )
