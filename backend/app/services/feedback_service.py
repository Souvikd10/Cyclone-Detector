"""
Stage 17 — Human-in-the-Loop: Feedback Service.

Persists forecaster accept/override actions, tracks override rates,
and provides a retrain trigger hook.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feedback import Feedback, FeedbackAction
from app.models.prediction import Prediction

logger = logging.getLogger(__name__)

# ── Retrain trigger threshold ──
OVERRIDE_THRESHOLD = 0.3  # If >30% of recent predictions are overridden, flag for retrain
RECENT_WINDOW = 20        # Consider last 20 feedbacks


async def submit_feedback(
    db: AsyncSession,
    storm_id: uuid.UUID,
    prediction_id: uuid.UUID,
    action: str,
    corrected_value: Optional[dict[str, Any]] = None,
    forecaster_note: Optional[str] = None,
    forecaster_id: Optional[str] = None,
) -> tuple[Feedback, bool]:
    """
    Record forecaster feedback on a prediction.

    Returns:
        Tuple of (feedback_record, retrain_triggered_flag)
    """
    # Validate prediction exists
    pred = await db.get(Prediction, prediction_id)
    if pred is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    # Create feedback record
    feedback = Feedback(
        id=uuid.uuid4(),
        prediction_id=prediction_id,
        storm_id=storm_id,
        action=FeedbackAction(action),
        corrected_value=corrected_value,
        forecaster_note=forecaster_note,
        forecaster_id=forecaster_id,
        created_at=datetime.now(timezone.utc),
    )

    db.add(feedback)
    await db.flush()

    # Log the action
    if action == "override":
        logger.warning(
            f"OVERRIDE on prediction {prediction_id} by {forecaster_id or 'unknown'}. "
            f"Note: {forecaster_note or 'none'}"
        )
    else:
        logger.info(
            f"ACCEPT on prediction {prediction_id} by {forecaster_id or 'unknown'}"
        )

    # Check if retrain should be triggered
    retrain_triggered = await _check_retrain_trigger(db, storm_id)

    return feedback, retrain_triggered


async def _check_retrain_trigger(
    db: AsyncSession,
    storm_id: uuid.UUID,
) -> bool:
    """
    Check if the override rate exceeds the threshold for the recent window.
    If so, log a retrain trigger event.

    In production, this would enqueue a Prefect flow for retraining.
    Currently just logs the intent.
    """
    # Get recent feedback for this storm
    result = await db.execute(
        select(Feedback)
        .where(Feedback.storm_id == storm_id)
        .order_by(Feedback.created_at.desc())
        .limit(RECENT_WINDOW)
    )
    recent = result.scalars().all()

    if len(recent) < 5:
        # Not enough data to make a retrain decision
        return False

    overrides = sum(1 for f in recent if f.action == FeedbackAction.override)
    override_rate = overrides / len(recent)

    if override_rate > OVERRIDE_THRESHOLD:
        logger.critical(
            f"RETRAIN TRIGGER: Storm {storm_id} — override rate {override_rate:.1%} "
            f"({overrides}/{len(recent)}) exceeds threshold {OVERRIDE_THRESHOLD:.0%}. "
            f"TODO: Enqueue Prefect retraining flow."
        )
        # In production: trigger Prefect flow here
        # await trigger_retrain_flow(storm_id, override_rate)
        return True

    return False


async def get_feedback_stats(
    db: AsyncSession,
    storm_id: uuid.UUID,
) -> dict[str, Any]:
    """
    Get accept/override statistics for a storm.
    """
    total_result = await db.execute(
        select(func.count(Feedback.id))
        .where(Feedback.storm_id == storm_id)
    )
    total = total_result.scalar() or 0

    override_result = await db.execute(
        select(func.count(Feedback.id))
        .where(
            Feedback.storm_id == storm_id,
            Feedback.action == FeedbackAction.override,
        )
    )
    overrides = override_result.scalar() or 0

    accepts = total - overrides
    override_rate = overrides / total if total > 0 else 0.0

    return {
        "storm_id": storm_id,
        "total": total,
        "accepts": accepts,
        "overrides": overrides,
        "override_rate": round(override_rate, 4),
    }
