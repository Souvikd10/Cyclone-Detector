"""
Pydantic schemas for Feedback endpoints (Stage 17).
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Any, Optional


class FeedbackCreate(BaseModel):
    """POST body for /storms/{id}/feedback."""
    prediction_id: UUID = Field(..., description="ID of the prediction being reviewed")
    action: str = Field(
        ...,
        pattern="^(accept|override)$",
        description="accept or override"
    )
    corrected_value: Optional[dict[str, Any]] = Field(
        None, description="Corrected prediction fields (only for override)"
    )
    forecaster_note: Optional[str] = Field(
        None, description="Free-text note from the forecaster"
    )
    forecaster_id: Optional[str] = Field(
        None, description="Identifier of the forecaster"
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    id: UUID
    prediction_id: UUID
    storm_id: UUID
    action: str
    corrected_value: Optional[dict[str, Any]] = None
    forecaster_note: Optional[str] = None
    forecaster_id: Optional[str] = None
    created_at: datetime
    retrain_triggered: bool = Field(
        False, description="Whether this feedback triggered a retraining check"
    )

    model_config = {"from_attributes": True}


class FeedbackListResponse(BaseModel):
    """List of feedback for a storm."""
    storm_id: UUID
    count: int
    feedbacks: list[FeedbackResponse]


class FeedbackStats(BaseModel):
    """Accept/override statistics for a storm."""
    storm_id: UUID
    total: int
    accepts: int
    overrides: int
    override_rate: float
