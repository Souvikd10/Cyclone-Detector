"""
Seed data router:
  POST /seed — populate the database with realistic mock data for development
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import get_db
from app.models.storm import Storm, StormStatus
from app.models.prediction import Prediction
from app.core.mock_data import generate_all_mock_data, reset_in_memory_data
from app.core.websocket_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Seed Data"])


@router.post("/seed")
async def seed_mock_data(
    clear: bool = True,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Populate the database with realistic mock storm and prediction data.
    """
    if db is not None:
        try:
            if clear:
                await db.execute(text("DELETE FROM feedback"))
                await db.execute(text("DELETE FROM predictions"))
                await db.execute(text("DELETE FROM storms"))
                await db.flush()
                logger.info("Cleared existing DB data")

            data = generate_all_mock_data()

            storm_count = 0
            for storm_data in data["storms"]:
                storm = Storm(
                    id=storm_data["id"],
                    name=storm_data["name"],
                    basin=storm_data["basin"],
                    status=StormStatus(storm_data["status"]),
                    category=storm_data["category"],
                    current_lat=storm_data["current_lat"],
                    current_lon=storm_data["current_lon"],
                    max_wind_speed=storm_data["max_wind_speed"],
                    min_pressure=storm_data["min_pressure"],
                    formed_at=storm_data["formed_at"],
                    last_updated=storm_data["last_updated"],
                    created_at=storm_data["created_at"],
                )
                db.add(storm)
                storm_count += 1

            await db.flush()

            prediction_count = 0
            for pred_data in data["predictions"]:
                prediction = Prediction(
                    id=pred_data["id"],
                    storm_id=pred_data["storm_id"],
                    timestamp=pred_data["timestamp"],
                    detection=pred_data["detection"],
                    structure=pred_data["structure"],
                    current_state=pred_data["current_state"],
                    forecast=pred_data["forecast"],
                    ri=pred_data["ri"],
                    uncertainty=pred_data["uncertainty"],
                    xai=pred_data["xai"],
                    impact=pred_data["impact"],
                    model_run_id=pred_data["model_run_id"],
                    created_at=pred_data["created_at"],
                )
                db.add(prediction)
                prediction_count += 1

            await db.flush()

            return {
                "status": "success",
                "storms_created": storm_count,
                "predictions_created": prediction_count,
                "storm_ids": [str(s["id"]) for s in data["storms"]],
                "message": "Mock data seeded successfully into PostgreSQL database.",
            }
        except Exception as e:
            logger.warning(f"Error seeding DB: {e}. Resetting in-memory store instead.")

    # Reset in-memory store
    counts = reset_in_memory_data()
    from app.core.mock_data import get_mock_storms
    seeded_storms = get_mock_storms()
    return {
        "status": "success",
        "storms_created": counts["storms_seeded"],
        "predictions_created": counts["predictions_seeded"],
        "storm_ids": [str(s["id"]) for s in seeded_storms],
        "message": "Mock data seeded successfully in standalone memory store.",
    }
