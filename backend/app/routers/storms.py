"""
Storm API routers:
  GET /storms               — list active/historical storms
  GET /storms/{id}/latest    — latest prediction for a storm
  GET /storms/{id}/history   — time series of predictions
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.storm import Storm, StormStatus
from app.models.prediction import Prediction
from app.schemas.storm import StormListItem, StormListResponse, StormDetail
from app.schemas.prediction import PredictionResponse, PredictionHistoryResponse
from app.core.mock_data import (
    get_mock_storms,
    get_mock_storm,
    get_mock_latest_prediction,
    get_mock_prediction_history,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Storms"])


@router.get("/storms", response_model=StormListResponse)
async def list_storms(
    status: str | None = Query(None, description="Filter by status: active, historical, dissipated"),
    db: AsyncSession | None = Depends(get_db),
):
    """
    List all tracked storms.
    Optionally filter by status.
    """
    if status:
        try:
            status_enum = StormStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of: active, historical, dissipated"
            )

    if db is not None:
        try:
            query = select(Storm).order_by(Storm.last_updated.desc())
            if status:
                query = query.where(Storm.status == status_enum)

            result = await db.execute(query)
            storms = result.scalars().all()
            if storms:
                return StormListResponse(
                    count=len(storms),
                    storms=[
                        StormListItem(
                            id=s.id,
                            name=s.name,
                            status=s.status.value,
                            category=s.category,
                            current_lat=s.current_lat,
                            current_lon=s.current_lon,
                            max_wind_speed=s.max_wind_speed,
                            min_pressure=s.min_pressure,
                            last_updated=s.last_updated,
                        )
                        for s in storms
                    ],
                )
        except Exception as e:
            logger.warning(f"Error querying DB for storms: {e}. Using mock store.")

    # Mock fallback
    mock_storms = get_mock_storms(status)
    return StormListResponse(
        count=len(mock_storms),
        storms=[
            StormListItem(
                id=s["id"],
                name=s["name"],
                status=s["status"],
                category=s["category"],
                current_lat=s["current_lat"],
                current_lon=s["current_lon"],
                max_wind_speed=s["max_wind_speed"],
                min_pressure=s["min_pressure"],
                last_updated=s["last_updated"],
            )
            for s in mock_storms
        ],
    )


@router.get("/storms/{storm_id}", response_model=StormDetail)
async def get_storm(
    storm_id: str,
    db: AsyncSession | None = Depends(get_db),
):
    """Get full details of a single storm."""
    if db is not None:
        try:
            storm = await db.get(Storm, UUID(storm_id))
            if storm:
                return StormDetail(
                    id=storm.id,
                    name=storm.name,
                    basin=storm.basin,
                    status=storm.status.value,
                    category=storm.category,
                    current_lat=storm.current_lat,
                    current_lon=storm.current_lon,
                    max_wind_speed=storm.max_wind_speed,
                    min_pressure=storm.min_pressure,
                    formed_at=storm.formed_at,
                    last_updated=storm.last_updated,
                    created_at=storm.created_at,
                )
        except Exception:
            pass

    # Mock fallback
    m = get_mock_storm(storm_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"Storm {storm_id} not found")
    return StormDetail(
        id=m["id"],
        name=m["name"],
        basin=m["basin"],
        status=m["status"],
        category=m["category"],
        current_lat=m["current_lat"],
        current_lon=m["current_lon"],
        max_wind_speed=m["max_wind_speed"],
        min_pressure=m["min_pressure"],
        formed_at=m["formed_at"],
        last_updated=m["last_updated"],
        created_at=m["created_at"],
    )


@router.get("/storms/{storm_id}/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    storm_id: str,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Get the most recent prediction for a storm.
    Includes all AI sub-objects + Stage 14 impact data.
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
                return _prediction_to_response(prediction)
        except Exception:
            pass

    # Mock fallback
    p = get_mock_latest_prediction(storm_id)
    if not p:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for storm {storm_id}"
        )
    return PredictionResponse(**p)


@router.get("/storms/{storm_id}/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    storm_id: str,
    from_time: datetime | None = Query(None, alias="from", description="Start of time range (ISO 8601)"),
    to_time: datetime | None = Query(None, alias="to", description="End of time range (ISO 8601)"),
    limit: int = Query(50, ge=1, le=500, description="Max predictions to return"),
    db: AsyncSession | None = Depends(get_db),
):
    """
    Get the time series of predictions for replay mode.
    Supports from/to time-range filtering.
    """
    if db is not None:
        try:
            uid = UUID(storm_id)
            query = (
                select(Prediction)
                .where(Prediction.storm_id == uid)
                .order_by(Prediction.timestamp.asc())
                .limit(limit)
            )
            if from_time:
                query = query.where(Prediction.timestamp >= from_time)
            if to_time:
                query = query.where(Prediction.timestamp <= to_time)

            result = await db.execute(query)
            predictions = result.scalars().all()
            if predictions:
                return PredictionHistoryResponse(
                    storm_id=uid,
                    count=len(predictions),
                    predictions=[_prediction_to_response(p) for p in predictions],
                )
        except Exception:
            pass

    # Mock fallback
    history = get_mock_prediction_history(storm_id, from_time, to_time, limit)
    storm = get_mock_storm(storm_id)
    s_id = storm["id"] if storm else storm_id
    return PredictionHistoryResponse(
        storm_id=s_id if isinstance(s_id, UUID) else UUID(str(s_id)),
        count=len(history),
        predictions=[PredictionResponse(**h) for h in history],
    )


@router.post("/storms/ingest", response_model=PredictionResponse)
async def ingest_prediction(
    contract_data: dict,
    storm_name: str | None = None,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Ingest a Member A Contract JSON directly from the AI Engine.
    Enriches with Stage 14 Impact + Stage 16 Alert + DB persistence + WebSocket broadcast.
    """
    from app.services.ai_bridge import ingest_member_a_prediction
    try:
        prediction_dict = await ingest_member_a_prediction(
            contract=contract_data,
            db=db,
            storm_name=storm_name,
        )
        return PredictionResponse(**prediction_dict)
    except Exception as e:
        logger.error(f"Failed to ingest Member A prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid prediction contract JSON: {e}")


@router.post("/storms/{storm_id}/predict", response_model=PredictionResponse)
async def trigger_prediction(
    storm_id: str,
    timestamp: str | None = None,
    db: AsyncSession | None = Depends(get_db),
):
    """
    Trigger AI model inference or telemetry prediction for a given storm ID.
    """
    from app.services.ai_bridge import ingest_member_a_prediction
    from app.core.mock_data import get_mock_storm, generate_mock_prediction

    # Check if we have sample outputs matching this storm
    import json
    from pathlib import Path
    samples_dir = Path("sample_outputs")
    
    sample_file = None
    if samples_dir.exists():
        for f in samples_dir.glob("*.json"):
            if storm_id.lower() in f.stem.lower():
                sample_file = f
                break

    if sample_file and sample_file.exists():
        try:
            with open(sample_file) as sf:
                contract = json.load(sf)
            prediction_dict = await ingest_member_a_prediction(contract=contract, db=db)
            return PredictionResponse(**prediction_dict)
        except Exception as e:
            logger.warning(f"Error loading sample file {sample_file}: {e}")

    # Generate realistic prediction
    storm = get_mock_storm(storm_id)
    template = {
        "name": storm["name"] if storm else "Live Cyclone",
        "basin": "NIO",
        "category": storm["category"] if storm else "VSCS",
        "base_lat": storm["current_lat"] if storm else 15.0,
        "base_lon": storm["current_lon"] if storm else 85.0,
        "wind": storm["max_wind_speed"] if storm else 75,
        "pressure": storm["min_pressure"] if storm else 980,
        "target_lat": (storm["current_lat"] + 4.0) if storm else 19.0,
        "target_lon": (storm["current_lon"] + 1.5) if storm else 86.5,
    }
    
    try:
        s_uuid = UUID(storm_id)
    except ValueError:
        s_uuid = UUID(str(storm["id"])) if storm else UUID("00000000-0000-0000-0000-000000000001")

    pred = generate_mock_prediction(storm_id=s_uuid, storm_template=template)
    from app.core.mock_data import _IN_MEMORY_DATA
    _IN_MEMORY_DATA.setdefault("predictions", []).append(pred)
    
    # Broadcast live update
    from app.core.websocket_manager import manager
    await manager.broadcast(
        str(s_uuid),
        {
            "type": "prediction_update",
            "storm_id": str(s_uuid),
            "timestamp": pred["timestamp"].isoformat(),
            "data": {
                "id": str(pred["id"]),
                "category": pred["current_state"]["category"],
                "wind": pred["current_state"]["max_wind_knots"],
                "pressure": pred["current_state"]["min_pressure_hpa"],
                "lat": pred["current_state"]["lat"],
                "lon": pred["current_state"]["lon"],
            },
        },
    )
    
    return PredictionResponse(**pred)


def _prediction_to_response(prediction: Prediction) -> PredictionResponse:
    """Convert ORM Prediction to Pydantic response."""
    return PredictionResponse(
        id=prediction.id,
        storm_id=prediction.storm_id,
        timestamp=prediction.timestamp,
        detection=prediction.detection,
        structure=prediction.structure,
        current_state=prediction.current_state,
        forecast=prediction.forecast,
        ri=prediction.ri,
        uncertainty=prediction.uncertainty,
        xai=prediction.xai,
        impact=prediction.impact,
        model_run_id=prediction.model_run_id,
        created_at=prediction.created_at,
    )
