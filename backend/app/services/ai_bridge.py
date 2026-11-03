"""
AI Integration Bridge:
Connects Member A's AI/ML pipeline (Stages 1–13) with Member B's backend
(Stages 14–17), database storage, and real-time WebSocket distribution.
"""

import sys
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.storm import Storm, StormStatus
from app.models.prediction import Prediction
from app.services.impact import compute_impact
from app.services.alert_generator import generate_cap_xml
from app.core.websocket_manager import manager
from app.core.mock_data import _IN_MEMORY_DATA, MOCK_STORMS, _wind_to_category

logger = logging.getLogger(__name__)

# Try to import Member A's pipeline
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AI_ENGINE_AVAILABLE = False
try:
    import cyclone_ai  # noqa: F401
    AI_ENGINE_AVAILABLE = True
except Exception as e:
    logger.info(f"Cyclone AI direct PyTorch execution not loaded in this env ({e}). Using contract bridge.")


def format_contract_to_prediction(contract: dict[str, Any], storm_uuid: uuid.UUID) -> dict[str, Any]:
    """
    Format Member A's raw contract JSON into backend internal prediction schema.
    Computes Stage 14 impact and adds identifiers.
    """
    ts_str = contract.get("timestamp")
    if isinstance(ts_str, str):
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
    elif isinstance(ts_str, datetime):
        ts = ts_str
    else:
        ts = datetime.now(timezone.utc)

    # Extract current state
    curr = contract.get("current_state", {})
    detection = contract.get("detection", {})
    center = detection.get("center", {"lat": 15.0, "lon": 85.0})
    
    # Wind and pressure estimation
    category = curr.get("category", "CS")
    wind_kt = 65.0
    forecast = contract.get("forecast", {})
    if forecast.get("intensity") and len(forecast["intensity"]) > 0:
        first_int = forecast["intensity"][0]
        wind_kt = float(first_int.get("wind_kt", 65.0))
    
    current_state_formatted = {
        "lat": center.get("lat", 15.0),
        "lon": center.get("lon", 85.0),
        "max_wind_knots": wind_kt,
        "min_pressure_hpa": round(1010 - wind_kt * 0.5, 1),
        "category": category,
        "dvorak_ci": 4.5,
        "motion_speed_knots": 12.0,
        "motion_direction_deg": 330.0,
    }

    # Format forecast track for Stage 14 Impact Engine
    track_raw = forecast.get("track", [])
    track_formatted = []
    intensity_formatted = []
    
    from datetime import timedelta
    for i, pt in enumerate(track_raw):
        t_offset = pt.get("t", f"+{i*6}h")
        try:
            hours = int(str(t_offset).replace("+", "").replace("h", ""))
        except Exception:
            hours = (i + 1) * 6
        pt_ts = ts + timedelta(hours=hours)
        pt_lat = float(pt.get("lat", 15.0))
        pt_lon = float(pt.get("lon", 85.0))
        
        # Match intensity if available
        int_pt = forecast.get("intensity", [])[i] if i < len(forecast.get("intensity", [])) else {}
        w = float(int_pt.get("wind_kt", wind_kt))
        
        track_formatted.append({
            "lat": pt_lat,
            "lon": pt_lon,
            "timestamp": pt_ts.isoformat(),
            "wind_speed": w,
            "pressure": round(1010 - w * 0.5, 1),
            "category": _wind_to_category(w),
        })
        
        intensity_formatted.append({
            "timestamp": pt_ts.isoformat(),
            "wind_speed": w,
            "pressure": round(1010 - w * 0.5, 1),
            "category": _wind_to_category(w),
        })

    forecast_formatted = {
        "track": track_formatted,
        "intensity": intensity_formatted,
    }

    # Compute Stage 14 Landfall Impact
    impact = compute_impact(forecast_formatted, current_state_formatted)

    # Uncertainty cone formatting
    uncertainty_raw = contract.get("uncertainty", {})
    spread_km = float(uncertainty_raw.get("novelty_score", 1.0)) * 60.0
    
    # Structure formatting
    struct_raw = contract.get("structure", {})
    structure_formatted = {
        "eye_detected": struct_raw.get("eye_present", False),
        "eye_radius_km": struct_raw.get("eye_diameter_km", 30.0) / 2.0 if struct_raw.get("eye_diameter_km") else None,
        "eyewall_completeness": struct_raw.get("eye_confidence", 0.8),
        "rain_bands": 4,
        "symmetry_score": struct_raw.get("symmetry", 0.75),
    }

    # Detection formatting
    bbox_raw = detection.get("bbox", [12.0, 82.0, 18.0, 88.0])
    detection_formatted = {
        "detected": detection.get("present", True),
        "confidence": detection.get("confidence", 0.95),
        "center": detection.get("center", {"lat": 15.0, "lon": 85.0}),
        "bounding_box": {
            "min_lat": bbox_raw[0] if len(bbox_raw) > 0 else 12.0,
            "min_lon": bbox_raw[1] if len(bbox_raw) > 1 else 82.0,
            "max_lat": bbox_raw[2] if len(bbox_raw) > 2 else 18.0,
            "max_lon": bbox_raw[3] if len(bbox_raw) > 3 else 88.0,
        },
    }

    # XAI formatting
    xai_raw = contract.get("xai", {})
    xai_formatted = {
        "feature_importance": {
            "sea_surface_temp": 0.35,
            "vertical_wind_shear": 0.25,
            "ocean_heat_content": 0.20,
            "upper_divergence": 0.12,
            "relative_humidity": 0.08,
        },
        "attention_map_url": xai_raw.get("gradcam_url"),
        "reasoning": f"AI Explanation Tags: {', '.join(xai_raw.get('explanation_tags', []))}",
    }

    # RI formatting
    ri_raw = contract.get("ri", {})
    ri_formatted = {
        "probability": ri_raw.get("probability", 0.15),
        "expected_increase_knots": 25.0 if ri_raw.get("probability", 0) > 0.5 else 10.0,
        "timeframe_hours": 24,
    }

    return {
        "id": uuid.uuid4(),
        "storm_id": storm_uuid,
        "timestamp": ts,
        "detection": detection_formatted,
        "structure": structure_formatted,
        "current_state": current_state_formatted,
        "forecast": forecast_formatted,
        "ri": ri_formatted,
        "uncertainty": {
            "track_spread_km": spread_km,
            "intensity_spread_knots": 12.0,
            "ensemble_members": 21,
        },
        "xai": xai_formatted,
        "impact": impact,
        "model_run_id": f"cyclone-ai-{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc),
    }


async def ingest_member_a_prediction(
    contract: dict[str, Any],
    db: Optional[AsyncSession] = None,
    storm_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Ingest a Member A contract JSON:
    1. Resolve or create Storm record
    2. Format into Prediction with Stage 14 impact
    3. Save to DB (if available) + in-memory store
    4. Broadcast update over WebSocket
    """
    raw_storm_id = str(contract.get("storm_id", "amphan"))
    name = storm_name or f"Cyclone {raw_storm_id.capitalize()}"
    
    # Generate deterministic UUID for known string IDs
    try:
        storm_uuid = uuid.UUID(raw_storm_id)
    except ValueError:
        storm_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, raw_storm_id)

    prediction_dict = format_contract_to_prediction(contract, storm_uuid)
    curr = prediction_dict["current_state"]

    # In-memory store sync
    existing_storm = next((s for s in _IN_MEMORY_DATA.get("storms", []) if str(s["id"]) == str(storm_uuid)), None)
    if not existing_storm:
        storm_entry = {
            "id": storm_uuid,
            "name": name,
            "basin": "NIO",
            "status": "active",
            "category": curr["category"],
            "current_lat": curr["lat"],
            "current_lon": curr["lon"],
            "max_wind_speed": curr["max_wind_knots"],
            "min_pressure": curr["min_pressure_hpa"],
            "formed_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        }
        _IN_MEMORY_DATA.setdefault("storms", []).append(storm_entry)
    else:
        existing_storm["current_lat"] = curr["lat"]
        existing_storm["current_lon"] = curr["lon"]
        existing_storm["max_wind_speed"] = curr["max_wind_knots"]
        existing_storm["min_pressure"] = curr["min_pressure_hpa"]
        existing_storm["category"] = curr["category"]
        existing_storm["last_updated"] = datetime.now(timezone.utc)

    _IN_MEMORY_DATA.setdefault("predictions", []).append(prediction_dict)

    # Database sync (if DB connected)
    if db is not None:
        try:
            db_storm = await db.get(Storm, storm_uuid)
            if not db_storm:
                db_storm = Storm(
                    id=storm_uuid,
                    name=name,
                    basin="NIO",
                    status=StormStatus.ACTIVE,
                    category=curr["category"],
                    current_lat=curr["lat"],
                    current_lon=curr["lon"],
                    max_wind_speed=curr["max_wind_knots"],
                    min_pressure=curr["min_pressure_hpa"],
                    formed_at=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )
                db.add(db_storm)
            else:
                db_storm.current_lat = curr["lat"]
                db_storm.current_lon = curr["lon"]
                db_storm.max_wind_speed = curr["max_wind_knots"]
                db_storm.min_pressure = curr["min_pressure_hpa"]
                db_storm.category = curr["category"]
                db_storm.last_updated = datetime.now(timezone.utc)

            await db.flush()

            db_prediction = Prediction(
                id=prediction_dict["id"],
                storm_id=storm_uuid,
                timestamp=prediction_dict["timestamp"],
                detection=prediction_dict["detection"],
                structure=prediction_dict["structure"],
                current_state=prediction_dict["current_state"],
                forecast=prediction_dict["forecast"],
                ri=prediction_dict["ri"],
                uncertainty=prediction_dict["uncertainty"],
                xai=prediction_dict["xai"],
                impact=prediction_dict["impact"],
                model_run_id=prediction_dict["model_run_id"],
                created_at=prediction_dict["created_at"],
            )
            db.add(db_prediction)
            await db.flush()
        except Exception as e:
            logger.warning(f"Error persisting AI prediction to DB: {e}")

    # WebSocket Broadcast
    try:
        await manager.broadcast(
            str(storm_uuid),
            {
                "type": "prediction_update",
                "storm_id": str(storm_uuid),
                "timestamp": prediction_dict["timestamp"].isoformat(),
                "data": {
                    "id": str(prediction_dict["id"]),
                    "category": curr["category"],
                    "wind": curr["max_wind_knots"],
                    "pressure": curr["min_pressure_hpa"],
                    "lat": curr["lat"],
                    "lon": curr["lon"],
                },
            },
        )
    except Exception as e:
        logger.warning(f"WebSocket broadcast error: {e}")

    return prediction_dict
