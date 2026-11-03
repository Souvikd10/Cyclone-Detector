"""
Impact API router — Stage 14:
  GET /storms/{id}/impact — landfall location/ETA + impact zone GeoJSON
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.storm import Storm
from app.models.prediction import Prediction
from app.services.impact import compute_impact
from app.schemas.impact import ImpactResponse
from app.core.mock_data import get_mock_latest_prediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Impact (Stage 14)"])


@router.get("/storms/{storm_id}/impact", response_model=ImpactResponse)
async def get_storm_impact(
    storm_id: str,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Stage 14 — Decision-Support Layer.

    Returns landfall estimation (location + ETA) and coastal impact zone GeoJSON,
    computed from the latest prediction's forecast track + intensity.
    """
    if db is not None:
        try:
            uid = UUID(storm_id)
            result = await db.execute(
                select(Prediction)
                .where(Prediction.storm_id == uid)
                .order_by(Prediction.timestamp.desc())
                .limit(1)
            )
            prediction = result.scalar_one_or_none()

            if prediction:
                if prediction.impact and prediction.impact.get("landfall_location"):
                    impact_data = prediction.impact
                else:
                    forecast = prediction.forecast or {}
                    current_state = prediction.current_state or {}
                    impact_data = compute_impact(forecast, current_state)
                    prediction.impact = impact_data
                    await db.flush()

                landfall_loc = impact_data.get("landfall_location")
                impact_zone = impact_data.get("impact_zone_geojson")

                return ImpactResponse(
                    storm_id=uid,
                    prediction_id=prediction.id,
                    landfall_location=landfall_loc,
                    landfall_eta=impact_data.get("landfall_eta"),
                    impact_zone=impact_zone,
                    wind_radius_km=(
                        impact_zone.get("properties", {}).get("wind_radius_km")
                        if impact_zone else None
                    ),
                    storm_surge_risk=(
                        impact_zone.get("properties", {}).get("storm_surge_risk")
                        if impact_zone else None
                    ),
                    population_exposed=None,
                    computed_at=datetime.now(timezone.utc),
                )
        except Exception:
            pass

    # Mock fallback
    p = get_mock_latest_prediction(storm_id)
    if not p:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for storm {storm_id}"
        )

    impact_data = p.get("impact") or compute_impact(p.get("forecast", {}), p.get("current_state", {}))
    landfall_loc = impact_data.get("landfall_location")
    impact_zone = impact_data.get("impact_zone_geojson")

    s_id = p["storm_id"]
    pred_id = p["id"]

    return ImpactResponse(
        storm_id=s_id if isinstance(s_id, UUID) else UUID(str(s_id)),
        prediction_id=pred_id if isinstance(pred_id, UUID) else UUID(str(pred_id)),
        landfall_location=landfall_loc,
        landfall_eta=impact_data.get("landfall_eta"),
        impact_zone=impact_zone,
        wind_radius_km=(
            impact_zone.get("properties", {}).get("wind_radius_km")
            if impact_zone else None
        ),
        storm_surge_risk=(
            impact_zone.get("properties", {}).get("storm_surge_risk")
            if impact_zone else None
        ),
        population_exposed=None,
        computed_at=datetime.now(timezone.utc),
    )
