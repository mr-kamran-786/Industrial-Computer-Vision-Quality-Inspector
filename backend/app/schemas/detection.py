from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    severity: Optional[str] = None
    threshold: Optional[float] = None
    bbox: BoundingBox


class Traceability(BaseModel):
    """Physical-part and production-context metadata for an inspection."""

    serial_number: Optional[str] = None
    batch_lot: Optional[str] = None
    production_line: Optional[str] = None
    station_id: Optional[str] = None
    operator_id: Optional[str] = None
    shift: Optional[str] = None


class PredictionResponse(BaseModel):
    inspection_id: Optional[str] = None
    filename: str
    model_key: str
    model_name: str
    model_version: str
    total_detections: int
    is_defective: bool
    quality_status: str
    highest_severity: Optional[str] = None
    decision_reason: Optional[str] = None
    inference_ms: Optional[float] = None
    detections: List[Detection]
    traceability: Traceability = Field(default_factory=Traceability)
    result_image_url: Optional[str] = None


class InspectionSummary(BaseModel):
    inspection_id: str
    original_filename: str
    quality_status: str
    is_defective: bool
    total_detections: int
    highest_severity: Optional[str] = None
    serial_number: Optional[str] = None
    production_line: Optional[str] = None
    reviewed: bool = False
    review_is_correct: Optional[bool] = None
    result_image_url: str
    created_at: datetime


class InspectionDetail(BaseModel):
    inspection_id: str
    original_filename: str
    saved_filename: str
    result_filename: str
    result_image_url: str
    quality_status: str
    is_defective: bool
    total_detections: int
    highest_severity: Optional[str] = None
    decision_reason: Optional[str] = None
    model_key: str
    model_name: str
    model_version: str
    inference_ms: Optional[float] = None
    traceability: Traceability = Field(default_factory=Traceability)
    reviewed: bool = False
    review_verdict: Optional[str] = None
    review_is_correct: Optional[bool] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    created_at: datetime
    detections: List[Detection]
