from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Operator ground-truth judgement for an inspection.

    ``verdict`` is the human's PASS/FAIL call; comparing it to the AI verdict is
    what lets us measure true escape rate and false-call rate.
    """

    verdict: str = Field(..., pattern="^(PASS|FAIL)$")
    reviewed_by: str = Field(..., min_length=1, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=2000)


class ReviewResponse(BaseModel):
    inspection_id: str
    quality_status: str
    review_verdict: str
    review_is_correct: bool
    reviewed_by: str
    reviewed_at: datetime
    review_notes: Optional[str] = None
