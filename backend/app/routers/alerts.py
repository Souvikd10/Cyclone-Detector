"""
Alert API router — Stage 16:
  GET /alerts/{storm_id} — CAP XML for current risk level
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.storm import Storm
from app.models.prediction import Prediction
from app.services.alert_generator import generate_cap_xml
from app.schemas.alert import AlertResponse, AlertInfo
from app.core.mock_data import get_mock_storm, get_mock_latest_prediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Alerts (Stage 16)"])


@router.get("/alerts/{storm_id}")
async def get_storm_alert(
    storm_id: str,
    format: str | None = None,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Stage 16 — Alert Dissemination.

    Generates a CAP 1.2 XML alert from the current risk/forecast data.

    Query params:
      - format=xml → returns raw XML (Content-Type: application/xml)
      - format=json or omitted → returns JSON wrapper with metadata + XML string

    NOTE: All alerts carry status=Test. This is a prototype only.
    """
    storm_name = "Cyclone"
    current_state = None
    forecast = {}
    impact = None
    detection = None

    if db is not None:
        try:
            uid = UUID(storm_id)
            storm = await db.get(Storm, uid)
            if storm:
                storm_name = storm.name
                result = await db.execute(
                    select(Prediction)
                    .where(Prediction.storm_id == uid)
                    .order_by(Prediction.timestamp.desc())
                    .limit(1)
                )
                prediction = result.scalar_one_or_none()
                if prediction:
                    current_state = prediction.current_state or {
                        "max_wind_knots": storm.max_wind_speed or 50,
                        "min_pressure_hpa": storm.min_pressure or 990,
                        "category": storm.category or "CS",
                        "lat": storm.current_lat,
                        "lon": storm.current_lon,
                    }
                    forecast = prediction.forecast or {}
                    impact = prediction.impact
                    detection = prediction.detection
        except Exception:
            pass

    # Mock fallback if DB didn't provide data
    if current_state is None:
        m_storm = get_mock_storm(storm_id)
        if not m_storm:
            raise HTTPException(status_code=404, detail=f"Storm {storm_id} not found")
        storm_name = m_storm["name"]
        m_pred = get_mock_latest_prediction(storm_id)
        if not m_pred:
            raise HTTPException(status_code=404, detail=f"No predictions found for storm {storm_id}")

        current_state = m_pred.get("current_state", {
            "max_wind_knots": m_storm.get("max_wind_speed", 65),
            "min_pressure_hpa": m_storm.get("min_pressure", 985),
            "category": m_storm.get("category", "VSCS"),
            "lat": m_storm["current_lat"],
            "lon": m_storm["current_lon"],
        })
        forecast = m_pred.get("forecast", {})
        impact = m_pred.get("impact")
        detection = m_pred.get("detection")

    cap_xml, metadata = generate_cap_xml(
        storm_id=str(storm_id),
        storm_name=storm_name,
        current_state=current_state,
        forecast=forecast,
        impact=impact,
        detection=detection,
    )

    # Return raw XML if requested
    if format == "xml":
        return Response(
            content=cap_xml,
            media_type="application/xml",
            headers={"Content-Disposition": f'inline; filename="alert-{storm_id}.xml"'},
        )

    # Return JSON wrapper by default
    return AlertResponse(
        info=AlertInfo(**metadata),
        cap_xml=cap_xml,
    )
