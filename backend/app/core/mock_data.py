"""
Realistic mock data generator for development.
Generates storms, predictions, and impact data that looks like real North Indian Ocean cyclones.
"""

import uuid
import random
from datetime import datetime, timedelta, timezone
from typing import Any


# ── Comprehensive Bay of Bengal & Arabian Sea (Indian Ocean) Cyclones ──
MOCK_STORMS: list[dict[str, Any]] = [
    # ── Bay of Bengal (BOB) Cyclones ──
    {
        "name": "Cyclone Amphan",
        "basin": "BOB",
        "category": "SuCS",  # Super Cyclonic Storm
        "base_lat": 16.0,
        "base_lon": 85.2,
        "wind": 140,
        "pressure": 907,
        "target_lat": 21.8,
        "target_lon": 88.3,
    },
    {
        "name": "Cyclone Remal",
        "basin": "BOB",
        "category": "SCS",   # Severe Cyclonic Storm
        "base_lat": 18.5,
        "base_lon": 88.0,
        "wind": 80,
        "pressure": 978,
        "target_lat": 21.7,
        "target_lon": 89.5,
    },
    {
        "name": "Cyclone Fani",
        "basin": "BOB",
        "category": "ESCS",  # Extremely Severe Cyclonic Storm
        "base_lat": 13.5,
        "base_lon": 83.6,
        "wind": 115,
        "pressure": 932,
        "target_lat": 19.8,
        "target_lon": 85.8,
    },
    {
        "name": "Cyclone Mocha",
        "basin": "BOB",
        "category": "SuCS",  # Super Cyclonic Storm
        "base_lat": 17.5,
        "base_lon": 89.5,
        "wind": 130,
        "pressure": 918,
        "target_lat": 20.1,
        "target_lon": 92.9,
    },
    {
        "name": "Cyclone Michaung",
        "basin": "BOB",
        "category": "SCS",
        "base_lat": 12.8,
        "base_lon": 81.2,
        "wind": 75,
        "pressure": 986,
        "target_lat": 15.8,
        "target_lon": 80.3,
    },
    {
        "name": "Cyclone Dana",
        "basin": "BOB",
        "category": "CS",
        "base_lat": 14.0,
        "base_lon": 86.5,
        "wind": 55,
        "pressure": 994,
        "target_lat": 20.8,
        "target_lon": 86.9,
    },

    # ── Arabian Sea & North Indian Ocean (AS/NIO) Cyclones ──
    {
        "name": "Cyclone Biparjoy",
        "basin": "AS",
        "category": "ESCS",
        "base_lat": 20.8,
        "base_lon": 66.5,
        "wind": 115,
        "pressure": 958,
        "target_lat": 23.2,
        "target_lon": 68.6,
    },
    {
        "name": "Cyclone Tauktae",
        "basin": "AS",
        "category": "ESCS",
        "base_lat": 18.5,
        "base_lon": 71.5,
        "wind": 120,
        "pressure": 950,
        "target_lat": 20.8,
        "target_lon": 71.1,
    },
    {
        "name": "Cyclone Nisarga",
        "basin": "AS",
        "category": "VSCS",
        "base_lat": 17.5,
        "base_lon": 72.5,
        "wind": 75,
        "pressure": 984,
        "target_lat": 18.3,
        "target_lon": 72.9,
    },
    {
        "name": "Cyclone Kyarr",
        "basin": "AS",
        "category": "SuCS",
        "base_lat": 17.5,
        "base_lon": 67.5,
        "wind": 135,
        "pressure": 915,
        "target_lat": 19.0,
        "target_lon": 63.0,
    },
    {
        "name": "Cyclone Maha",
        "basin": "AS",
        "category": "ESCS",
        "base_lat": 17.8,
        "base_lon": 65.5,
        "wind": 100,
        "pressure": 956,
        "target_lat": 20.0,
        "target_lon": 69.5,
    },
    {
        "name": "Cyclone Fengal",
        "basin": "AS",
        "category": "CS",
        "base_lat": 10.5,
        "base_lon": 82.0,
        "wind": 50,
        "pressure": 996,
        "target_lat": 12.0,
        "target_lon": 79.8,
    },
]


def _generate_track(
    start_lat: float,
    start_lon: float,
    target_lat: float,
    target_lon: float,
    base_time: datetime,
    hours: int = 120,
    step_hours: int = 6,
) -> list[dict[str, Any]]:
    """Generate a realistic-looking forecast track from start → target."""
    track = []
    steps = hours // step_hours
    for i in range(steps + 1):
        t = i / steps
        # Simple interpolation with slight curve and noise
        lat = start_lat + (target_lat - start_lat) * t + random.gauss(0, 0.15)
        lon = start_lon + (target_lon - start_lon) * t + random.gauss(0, 0.15)
        ts = base_time + timedelta(hours=i * step_hours)
        # Wind decreases near landfall in a realistic pattern
        wind_factor = 1.0 - 0.3 * max(0, t - 0.7) / 0.3
        base_wind = 80 + random.gauss(0, 5)
        track.append({
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "timestamp": ts.isoformat(),
            "wind_speed": round(base_wind * wind_factor, 1),
            "pressure": round(990 - base_wind * wind_factor * 0.4 + random.gauss(0, 2), 1),
            "category": _wind_to_category(base_wind * wind_factor),
        })
    return track


def _wind_to_category(wind_knots: float) -> str:
    """Convert wind speed (knots) to IMD category."""
    if wind_knots >= 120:
        return "SuCS"   # Super Cyclonic Storm
    elif wind_knots >= 90:
        return "ESCS"   # Extremely Severe
    elif wind_knots >= 64:
        return "VSCS"   # Very Severe
    elif wind_knots >= 48:
        return "SCS"    # Severe
    elif wind_knots >= 34:
        return "CS"     # Cyclonic Storm
    elif wind_knots >= 28:
        return "DD"     # Deep Depression
    else:
        return "D"      # Depression


def _generate_intensity(
    track: list[dict],
    base_wind: float,
    base_pressure: float,
) -> list[dict[str, Any]]:
    """Generate intensity forecast aligned with track."""
    return [
        {
            "timestamp": pt["timestamp"],
            "wind_speed": pt["wind_speed"],
            "pressure": pt["pressure"],
            "category": pt["category"],
        }
        for pt in track
    ]


def _generate_uncertainty_cone(track: list[dict]) -> dict[str, Any]:
    """Generate a simple uncertainty cone GeoJSON around the track."""
    if not track:
        return {}

    # Build cone as widening polygon around track
    left_side = []
    right_side = []

    for i, pt in enumerate(track):
        # Cone widens over time
        spread = 0.5 + i * 0.3
        left_side.append([pt["lon"] - spread, pt["lat"] + spread * 0.3])
        right_side.append([pt["lon"] + spread, pt["lat"] - spread * 0.3])

    # Close the polygon
    coords = left_side + list(reversed(right_side))
    coords.append(coords[0])  # close ring

    return {
        "cone_geojson": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {"type": "uncertainty_cone"},
        },
        "track_spread_km": round(50 + len(track) * 15, 1),
        "intensity_spread_knots": round(10 + random.gauss(0, 3), 1),
        "ensemble_members": 21,
    }


def _generate_impact(
    track: list[dict],
    wind_speed: float,
) -> dict[str, Any]:
    """Generate Stage 14 impact data — landfall location and impact zone."""
    if not track or len(track) < 3:
        return {}

    # Pick a point ~70% through the track as simulated landfall
    landfall_idx = int(len(track) * 0.7)
    landfall_pt = track[landfall_idx]

    # Wind radius scales with intensity
    wind_radius_deg = max(0.5, wind_speed / 100.0)

    # Build impact zone as a simple circle around landfall
    import math
    center_lon = landfall_pt["lon"]
    center_lat = landfall_pt["lat"]
    circle_points = []
    for angle in range(0, 360, 10):
        rad = math.radians(angle)
        circle_points.append([
            round(center_lon + wind_radius_deg * math.cos(rad), 4),
            round(center_lat + wind_radius_deg * math.sin(rad), 4),
        ])
    circle_points.append(circle_points[0])  # close ring

    return {
        "landfall_location": {
            "lat": landfall_pt["lat"],
            "lon": landfall_pt["lon"],
            "name": "Simulated Coastal Location",
            "country": "IN",
        },
        "landfall_eta": landfall_pt["timestamp"],
        "impact_zone_geojson": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [circle_points],
            },
            "properties": {
                "wind_category": _wind_to_category(wind_speed),
                "estimated_surge_m": round(max(0.5, wind_speed / 30.0), 1),
            },
        },
    }


def generate_mock_storm(
    template: dict | None = None,
    status: str = "active",
) -> dict[str, Any]:
    """Generate a single mock storm dict (ready for ORM insertion)."""
    if template is None:
        template = random.choice(MOCK_STORMS)

    storm_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    formed = now - timedelta(days=random.randint(1, 5))

    return {
        "id": storm_id,
        "name": template["name"],
        "basin": template["basin"],
        "status": status,
        "category": template["category"],
        "current_lat": template["base_lat"],
        "current_lon": template["base_lon"],
        "max_wind_speed": template["wind"],
        "min_pressure": template["pressure"],
        "formed_at": formed,
        "last_updated": now,
        "created_at": now,
    }


def generate_mock_prediction(
    storm_id: uuid.UUID,
    storm_template: dict,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Generate a full mock prediction with all sub-objects."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    track = _generate_track(
        start_lat=storm_template["base_lat"],
        start_lon=storm_template["base_lon"],
        target_lat=storm_template["target_lat"],
        target_lon=storm_template["target_lon"],
        base_time=timestamp,
    )

    intensity = _generate_intensity(
        track, storm_template["wind"], storm_template["pressure"]
    )

    uncertainty = _generate_uncertainty_cone(track)
    impact = _generate_impact(track, storm_template["wind"])

    return {
        "id": uuid.uuid4(),
        "storm_id": storm_id,
        "timestamp": timestamp,
        "detection": {
            "detected": True,
            "confidence": round(random.uniform(0.85, 0.99), 3),
            "bounding_box": {
                "min_lat": storm_template["base_lat"] - 2,
                "max_lat": storm_template["base_lat"] + 2,
                "min_lon": storm_template["base_lon"] - 2,
                "max_lon": storm_template["base_lon"] + 2,
            },
            "center": {
                "lat": storm_template["base_lat"],
                "lon": storm_template["base_lon"],
            },
        },
        "structure": {
            "eye_detected": storm_template["wind"] >= 64,
            "eye_radius_km": round(random.uniform(20, 50), 1) if storm_template["wind"] >= 64 else None,
            "eyewall_completeness": round(random.uniform(0.6, 1.0), 2) if storm_template["wind"] >= 64 else None,
            "rain_bands": random.randint(2, 6),
            "symmetry_score": round(random.uniform(0.4, 0.9), 2),
        },
        "current_state": {
            "lat": storm_template["base_lat"],
            "lon": storm_template["base_lon"],
            "max_wind_knots": storm_template["wind"],
            "min_pressure_hpa": storm_template["pressure"],
            "category": storm_template["category"],
            "dvorak_ci": round(random.uniform(3.0, 7.0), 1),
            "motion_speed_knots": round(random.uniform(8, 18), 1),
            "motion_direction_deg": round(random.uniform(300, 350), 1),
        },
        "forecast": {
            "track": track,
            "intensity": intensity,
        },
        "ri": {
            "probability": round(random.uniform(0.05, 0.45), 3),
            "expected_increase_knots": round(random.uniform(10, 35), 1),
            "timeframe_hours": 24,
        },
        "uncertainty": uncertainty,
        "xai": {
            "feature_importance": {
                "sea_surface_temp": round(random.uniform(0.15, 0.35), 3),
                "vertical_wind_shear": round(random.uniform(0.10, 0.25), 3),
                "ocean_heat_content": round(random.uniform(0.10, 0.20), 3),
                "upper_divergence": round(random.uniform(0.05, 0.15), 3),
                "relative_humidity": round(random.uniform(0.05, 0.12), 3),
            },
            "attention_map_url": None,
            "reasoning": (
                f"Storm shows {'well-organized' if storm_template['wind'] >= 64 else 'developing'} "
                f"convective structure. SST of ~29°C supports "
                f"{'further intensification' if storm_template['wind'] < 90 else 'maintenance of current intensity'}."
            ),
        },
        "impact": impact,
        "model_run_id": f"mock-run-{uuid.uuid4().hex[:8]}",
        "created_at": timestamp,
    }


def generate_all_mock_data() -> dict[str, list[dict]]:
    """
    Generate a full set of realistic cyclone data:
    - 3 active storms, 2 historical
    - 5 predictions per storm (every 6 hours going back 24h)
    """
    storms: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for i, template in enumerate(MOCK_STORMS):
        status = "active" if i < 3 else "historical"
        storm = generate_mock_storm(template, status)
        storms.append(storm)

        # Generate 5 predictions, 6 hours apart
        for j in range(5):
            ts = now - timedelta(hours=(4 - j) * 6)
            pred = generate_mock_prediction(storm["id"], template, ts)
            predictions.append(pred)

    return {"storms": storms, "predictions": predictions}


# ── In-Memory Store for Standalone Mode ──
# ── In-Memory Store for Standalone Mode ──
_IN_MEMORY_DATA: dict[str, list[dict[str, Any]]] = generate_all_mock_data()
_IN_MEMORY_FEEDBACKS: list[dict[str, Any]] = []


def reset_in_memory_data() -> dict[str, int]:
    """Regenerate fresh mock data in memory."""
    global _IN_MEMORY_DATA, _IN_MEMORY_FEEDBACKS
    _IN_MEMORY_DATA = generate_all_mock_data()
    _IN_MEMORY_FEEDBACKS = []
    return {
        "storms_seeded": len(_IN_MEMORY_DATA["storms"]),
        "predictions_seeded": len(_IN_MEMORY_DATA["predictions"]),
    }


def get_mock_storms(status: str | None = None) -> list[dict[str, Any]]:
    """Retrieve storms from in-memory store (auto-seeds if empty)."""
    global _IN_MEMORY_DATA
    if not _IN_MEMORY_DATA.get("storms"):
        _IN_MEMORY_DATA = generate_all_mock_data()
    storms = _IN_MEMORY_DATA.get("storms", [])
    if status:
        return [s for s in storms if s.get("status") == status]
    return storms


def get_mock_storm(storm_id: Any) -> dict[str, Any] | None:
    """Retrieve a single storm by ID or slug."""
    global _IN_MEMORY_DATA
    if not _IN_MEMORY_DATA.get("storms"):
        _IN_MEMORY_DATA = generate_all_mock_data()
    s_id = str(storm_id).lower()
    for s in _IN_MEMORY_DATA.get("storms", []):
        if str(s.get("id", "")).lower() == s_id:
            return s
        name_slug = s.get("name", "").lower().replace(" ", "-")
        if name_slug in s_id or s_id in name_slug:
            return s
    return None



def get_mock_latest_prediction(storm_id: Any) -> dict[str, Any] | None:
    """Retrieve latest prediction for a storm."""
    storm = get_mock_storm(storm_id)
    if not storm:
        return None
    preds = [p for p in _IN_MEMORY_DATA.get("predictions", []) if str(p.get("storm_id", "")) == str(storm["id"])]
    if preds:
        return sorted(preds, key=lambda x: x["timestamp"], reverse=True)[0]
    return None


def get_mock_prediction_history(
    storm_id: Any,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve prediction history for a storm."""
    storm = get_mock_storm(storm_id)
    if not storm:
        return []
    preds = [p for p in _IN_MEMORY_DATA.get("predictions", []) if str(p.get("storm_id", "")) == str(storm["id"])]
    if from_time:
        preds = [p for p in preds if p["timestamp"] >= from_time]
    if to_time:
        preds = [p for p in preds if p["timestamp"] <= to_time]
    return preds[:limit]


def add_mock_feedback(feedback_dict: dict[str, Any]) -> dict[str, Any]:
    """Record a forecaster feedback in memory."""
    entry = dict(feedback_dict)
    entry["id"] = uuid.uuid4()
    entry["created_at"] = datetime.now(timezone.utc)
    _IN_MEMORY_FEEDBACKS.append(entry)
    return entry
