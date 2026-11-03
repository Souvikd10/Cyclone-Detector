"""
Feedback API router — Stage 17:
  POST /storms/{id}/feedback — forecaster accept/override actions
"""

import logging
import uuid
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.storm import Storm
from app.schemas.feedback import FeedbackCreate, FeedbackResponse, FeedbackStats
from app.services.feedback_service import submit_feedback, get_feedback_stats
from app.core.mock_data import add_mock_feedback

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback (Stage 17)"])


@router.post("/storms/{storm_id}/feedback", response_model=FeedbackResponse, status_code=201)
async def post_feedback(
    storm_id: str,
    body: FeedbackCreate,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Stage 17 — Human-in-the-Loop.

    Record a forecaster's accept or override action on a prediction.
    If the override rate exceeds the threshold, flags for retraining.
    """
    if db is not None:
        try:
            uid = UUID(storm_id)
            feedback, retrain_triggered = await submit_feedback(
                db=db,
                storm_id=uid,
                prediction_id=body.prediction_id,
                action=body.action,
                corrected_value=body.corrected_value,
                forecaster_note=body.forecaster_note,
                forecaster_id=body.forecaster_id,
            )
            return FeedbackResponse(
                id=feedback.id,
                prediction_id=feedback.prediction_id,
                storm_id=feedback.storm_id,
                action=feedback.action.value,
                corrected_value=feedback.corrected_value,
                forecaster_note=feedback.forecaster_note,
                forecaster_id=feedback.forecaster_id,
                created_at=feedback.created_at,
                retrain_triggered=retrain_triggered,
            )
        except Exception as e:
            logger.warning(f"DB feedback recording error: {e}. Falling back to in-memory store.")

    # In-memory mock store
    feedback_entry = add_mock_feedback({
        "storm_id": storm_id,
        "prediction_id": body.prediction_id,
        "action": body.action,
        "corrected_value": body.corrected_value,
        "forecaster_note": body.forecaster_note,
        "forecaster_id": body.forecaster_id,
    })

    return FeedbackResponse(
        id=feedback_entry["id"],
        prediction_id=body.prediction_id,
        storm_id=UUID(str(storm_id)) if len(str(storm_id)) == 36 else uuid.uuid4(),
        action=body.action,
        corrected_value=body.corrected_value,
        forecaster_note=body.forecaster_note,
        forecaster_id=body.forecaster_id,
        created_at=feedback_entry["created_at"],
        retrain_triggered=False,
    )


@router.get("/storms/{storm_id}/feedback/stats", response_model=FeedbackStats)
async def get_storm_feedback_stats(
    storm_id: str,
    db: AsyncSession | None = Depends(get_db),
):
    """Get accept/override statistics for a storm."""
    if db is not None:
        try:
            uid = UUID(storm_id)
            stats = await get_feedback_stats(db, uid)
            return FeedbackStats(**stats)
        except Exception:
            pass

    try:
        s_uuid = UUID(storm_id)
    except Exception:
        s_uuid = uuid.uuid4()

    return FeedbackStats(
        storm_id=s_uuid,
        total=0,
        accepts=0,
        overrides=0,
        override_rate=0.0,
    )
