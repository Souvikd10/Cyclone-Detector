"""
Stage 14 — Decision-Support Layer: Landfall Estimation & Impact Zone.

Takes Member A's forecast track + intensity → estimates landfall location/ETA,
combines with coastline data (GeoPandas) → generates potential coastal impact zone GeoJSON.
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded geospatial data ──
_coastline_gdf = None
_land_gdf = None


def _load_coastline():
    """Load Natural Earth coastline shapefile via GeoPandas (lazy, cached)."""
    global _coastline_gdf
    if _coastline_gdf is not None:
        return _coastline_gdf

    try:
        import geopandas as gpd
        settings = get_settings()
        path = Path(settings.coastline_shapefile)
        if path.exists():
            _coastline_gdf = gpd.read_file(str(path))
            logger.info(f"Loaded coastline data: {len(_coastline_gdf)} features")
        else:
            logger.warning(f"Coastline shapefile not found at {path}. Using fallback.")
            _coastline_gdf = None
    except ImportError:
        logger.warning("GeoPandas not available — using analytical fallback for landfall estimation.")
        _coastline_gdf = None

    return _coastline_gdf


def _load_land():
    """Load Natural Earth land polygons."""
    global _land_gdf
    if _land_gdf is not None:
        return _land_gdf

    try:
        import geopandas as gpd
        settings = get_settings()
        path = Path(settings.land_shapefile)
        if path.exists():
            _land_gdf = gpd.read_file(str(path))
            logger.info(f"Loaded land data: {len(_land_gdf)} features")
        else:
            _land_gdf = None
    except ImportError:
        _land_gdf = None

    return _land_gdf


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _estimate_landfall_geospatial(
    track_points: list[dict],
) -> Optional[dict[str, Any]]:
    """
    Use GeoPandas to find where the forecast track intersects land.
    Returns landfall point dict or None if no landfall detected.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        land_gdf = _load_land()
        if land_gdf is None or len(track_points) < 2:
            return None

        # Build track as a LineString
        coords = [(pt["lon"], pt["lat"]) for pt in track_points]
        track_line = LineString(coords)

        # Dissolve all land into a single MultiPolygon for intersection
        land_union = land_gdf.geometry.union_all()

        # Find intersection
        intersection = track_line.intersection(land_union)

        if intersection.is_empty:
            return None

        # Get the first point of intersection
        if intersection.geom_type == "Point":
            landfall_point = intersection
        elif intersection.geom_type == "MultiPoint":
            landfall_point = intersection.geoms[0]
        elif intersection.geom_type in ("LineString", "MultiLineString"):
            # Track enters land — take the first coordinate
            if intersection.geom_type == "MultiLineString":
                first_line = intersection.geoms[0]
            else:
                first_line = intersection
            landfall_point = Point(first_line.coords[0])
        else:
            return None

        # Interpolate ETA by finding which track segment contains the landfall point
        landfall_lon, landfall_lat = landfall_point.x, landfall_point.y
        eta = _interpolate_eta(track_points, landfall_lat, landfall_lon)

        return {
            "lat": round(landfall_lat, 4),
            "lon": round(landfall_lon, 4),
            "eta": eta,
        }

    except Exception as e:
        logger.error(f"GeoPandas landfall estimation failed: {e}")
        return None


def _estimate_landfall_analytical(
    track_points: list[dict],
) -> Optional[dict[str, Any]]:
    """
    Fallback analytical landfall estimation for when GeoPandas is unavailable.
    Uses a simple bounding box for the Indian subcontinent coastline.
    """
    if len(track_points) < 2:
        return None

    # Simplified Indian coastline bounding — points that are roughly "on land"
    # This is a crude approximation; the real GeoPandas path is much better.
    for i, pt in enumerate(track_points):
        lat, lon = pt["lat"], pt["lon"]

        # Simple check: is this point roughly on Indian landmass?
        on_land = False

        # West coast of India (roughly)
        if 8.0 <= lat <= 23.5 and 68.0 <= lon <= 78.0:
            if lon < 73.0 + (lat - 8.0) * 0.3:  # rough west coast line
                on_land = True

        # East coast of India (roughly)
        if 8.0 <= lat <= 22.0 and 78.0 <= lon <= 88.0:
            if lon > 87.0 - (lat - 8.0) * 0.5:  # rough east coast line
                on_land = False  # still over Bay of Bengal mostly
            else:
                on_land = True

        # Bangladesh / Myanmar coast
        if 20.0 <= lat <= 26.0 and 88.0 <= lon <= 93.0:
            on_land = True

        if on_land:
            return {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "eta": pt.get("timestamp"),
            }

    return None


def _interpolate_eta(
    track_points: list[dict],
    target_lat: float,
    target_lon: float,
) -> Optional[str]:
    """Interpolate the ETA at a specific lat/lon along the track."""
    if not track_points:
        return None

    min_dist = float("inf")
    best_ts = track_points[0].get("timestamp")

    for i, pt in enumerate(track_points):
        d = _haversine_km(pt["lat"], pt["lon"], target_lat, target_lon)
        if d < min_dist:
            min_dist = d
            best_ts = pt.get("timestamp")

    return best_ts


def _build_impact_zone(
    lat: float,
    lon: float,
    wind_speed_knots: float,
) -> dict[str, Any]:
    """
    Build a GeoJSON impact zone polygon around the landfall point.
    Radius scales with wind speed.
    """
    # Wind radius in degrees (approximate): ~1 degree per 50 knots
    radius_deg = max(0.5, wind_speed_knots / 80.0)

    # Generate circle polygon
    points = []
    for angle in range(0, 360, 5):
        rad = math.radians(angle)
        # Adjust for latitude distortion
        lat_offset = radius_deg * math.sin(rad)
        lon_offset = radius_deg * math.cos(rad) / math.cos(math.radians(lat))
        points.append([
            round(lon + lon_offset, 5),
            round(lat + lat_offset, 5),
        ])
    points.append(points[0])  # close ring

    # Determine storm surge risk based on wind speed
    if wind_speed_knots >= 100:
        surge_risk = "extreme"
        surge_m = round(wind_speed_knots / 25.0, 1)
    elif wind_speed_knots >= 64:
        surge_risk = "high"
        surge_m = round(wind_speed_knots / 30.0, 1)
    elif wind_speed_knots >= 48:
        surge_risk = "moderate"
        surge_m = round(wind_speed_knots / 40.0, 1)
    else:
        surge_risk = "low"
        surge_m = round(max(0.3, wind_speed_knots / 50.0), 1)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [points],
        },
        "properties": {
            "wind_category": _wind_to_imd_category(wind_speed_knots),
            "storm_surge_risk": surge_risk,
            "estimated_surge_m": surge_m,
            "wind_radius_km": round(radius_deg * 111.0, 1),
        },
    }


def _wind_to_imd_category(wind_knots: float) -> str:
    """Convert wind speed (knots) to IMD category."""
    if wind_knots >= 120:
        return "SuCS"
    elif wind_knots >= 90:
        return "ESCS"
    elif wind_knots >= 64:
        return "VSCS"
    elif wind_knots >= 48:
        return "SCS"
    elif wind_knots >= 34:
        return "CS"
    elif wind_knots >= 28:
        return "DD"
    else:
        return "D"


def compute_impact(
    forecast_data: dict[str, Any],
    current_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main entry point for Stage 14 impact computation.

    Args:
        forecast_data: Member A's forecast dict with "track" and "intensity" arrays.
        current_state: Current storm state dict with "max_wind_knots" etc.

    Returns:
        Impact dict: {landfall_location, landfall_eta, impact_zone_geojson}
    """
    track_points = forecast_data.get("track", [])
    if not track_points:
        return {
            "landfall_location": None,
            "landfall_eta": None,
            "impact_zone_geojson": None,
        }

    # Try GeoPandas first, fall back to analytical
    landfall = _estimate_landfall_geospatial(track_points)
    if landfall is None:
        landfall = _estimate_landfall_analytical(track_points)

    if landfall is None:
        # No landfall predicted — storm stays over ocean
        return {
            "landfall_location": None,
            "landfall_eta": None,
            "impact_zone_geojson": None,
        }

    # Get wind speed for impact zone sizing
    wind_speed = 65.0  # default
    if current_state and "max_wind_knots" in current_state:
        wind_speed = current_state["max_wind_knots"]

    # Build impact zone
    impact_zone = _build_impact_zone(
        landfall["lat"], landfall["lon"], wind_speed
    )

    return {
        "landfall_location": {
            "lat": landfall["lat"],
            "lon": landfall["lon"],
            "name": None,  # Would need reverse geocoding for actual name
            "country": "IN",
        },
        "landfall_eta": landfall.get("eta"),
        "impact_zone_geojson": impact_zone,
    }
