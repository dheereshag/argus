"""Domain models, internal dataclasses, and thin API response schemas for Argus ANPR."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True, frozen=True)
class OCRToken:
    text: str
    score: float
    cx: float | None = None
    cy: float | None = None
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class PlateCandidate:
    y_pos: float
    rank: int
    info: dict[str, Any]
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectedVehicle:
    vehicle_type: str
    box: tuple[int, int, int, int]
    crop: Any = None
    crop_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectionResult:
    vehicles: list[DetectedVehicle] = field(default_factory=list)


class PlateResult(BaseModel):
    plate: str = Field(description="Normalized Indian vehicle registration number")
    execution_time_ms: float = Field(default=0.0, description="Processing duration in milliseconds")


class RecognitionResponse(BaseModel):
    results: list[PlateResult] = Field(default_factory=list, description="Plate result items")
    execution_time_ms: float = Field(..., description="Processing duration in milliseconds")


class APIErrorResponse(BaseModel):
    success: bool = Field(False)
    status_code: int = Field(...)
    message: str = Field(...)
    error_type: str = Field(...)
    details: Any = Field(None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
