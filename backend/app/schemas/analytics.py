from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class DefectCount(BaseModel):
    class_name: str
    count: int


class SeverityCount(BaseModel):
    severity: str
    count: int


class RecentFailure(BaseModel):
    inspection_id: str
    original_filename: str
    total_detections: int
    highest_severity: Optional[str] = None
    result_image_url: str
    created_at: datetime


class TrendPoint(BaseModel):
    date: str
    total: int
    pass_count: int
    fail_count: int
    defect_rate_percent: float
    dppm: float


class AnalyticsTrend(BaseModel):
    days: int
    points: List[TrendPoint]


class AnalyticsSummary(BaseModel):
    # Volume & yield
    total_inspections: int
    pass_count: int
    fail_count: int
    defect_rate_percent: float
    yield_percent: float
    dppm: float  # defective parts per million

    # Detection stats
    total_detected_defects: int
    average_confidence: Optional[float]
    most_common_defect: Optional[str]
    defect_distribution: List[DefectCount]
    severity_distribution: List[SeverityCount]

    # Ground-truth feedback (only meaningful once inspections are reviewed)
    reviewed_count: int
    escape_rate_percent: Optional[float]  # AI PASS but human FAIL
    false_call_rate_percent: Optional[float]  # AI FAIL but human PASS
    agreement_rate_percent: Optional[float]  # AI verdict == human verdict

    recent_failures: List[RecentFailure]
