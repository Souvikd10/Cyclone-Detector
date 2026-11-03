"""
Pydantic schemas for Prediction API responses.
Matches Member A's prediction JSON structure.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Any, Optional


class TrackPoint(BaseModel):
    """Single point on the forecast track."""
    lat: float
    lon: float
    timestamp: datetime
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    category: Optional[str] = None


class IntensityPoint(BaseModel):
    """Single point on the intensity forecast."""
    timestamp: datetime
    wind_speed: float
    pressure: float
    category: Optional[str] = None


class ForecastData(BaseModel):
    """Forecast track + intensity arrays."""
    track: list[TrackPoint] = []
    intensity: list[IntensityPoint] = []


class DetectionData(BaseModel):
    """Storm detection output."""
    detected: bool = True
    confidence: float = 0.95
    bounding_box: Optional[dict[str, float]] = None
    center: Optional[dict[str, float]] = None


class StructureData(BaseModel):
    """Storm structure analysis."""
    eye_detected: bool = False
    eye_radius_km: Optional[float] = None
    eyewall_completeness: Optional[float] = None
    rain_bands: Optional[int] = None
    symmetry_score: Optional[float] = None


class CurrentStateData(BaseModel):
    """Current storm state (Dvorak estimate, etc.)."""
    lat: float
    lon: float
    max_wind_knots: float
    min_pressure_hpa: float
    category: str
    dvorak_ci: Optional[float] = None
    motion_speed_knots: Optional[float] = None
    motion_direction_deg: Optional[float] = None


class RIData(BaseModel):
    """Rapid intensification prediction."""
    probability: float = 0.0
    expected_increase_knots: Optional[float] = None
    timeframe_hours: int = 24


class UncertaintyData(BaseModel):
    """Uncertainty cone / ensemble spread."""
    cone_geojson: Optional[dict[str, Any]] = None
    track_spread_km: Optional[float] = None
    intensity_spread_knots: Optional[float] = None
    ensemble_members: Optional[int] = None


class XAIData(BaseModel):
    """Explainability / attribution data."""
    feature_importance: Optional[dict[str, float]] = None
    attention_map_url: Optional[str] = None
    reasoning: Optional[str] = None


class ImpactData(BaseModel):
    """Stage 14: Landfall + impact zone computed by this backend."""
    landfall_location: Optional[dict[str, Any]] = Field(
        None, description='{"lat": ..., "lon": ..., "name": "..."}'
    )
    landfall_eta: Optional[datetime] = None
    impact_zone_geojson: Optional[dict[str, Any]] = None


class PredictionResponse(BaseModel):
    """Full prediction response — combines Member A's output + Stage 14 impact."""
    id: UUID
    storm_id: UUID
    timestamp: datetime
    detection: Optional[DetectionData] = None
    structure: Optional[StructureData] = None
    current_state: Optional[CurrentStateData] = None
    forecast: Optional[ForecastData] = None
    ri: Optional[RIData] = None
    uncertainty: Optional[UncertaintyData] = None
    xai: Optional[XAIData] = None
    impact: Optional[ImpactData] = None
    model_run_id: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionHistoryResponse(BaseModel):
    """Response for GET /storms/{id}/history — time series of predictions."""
    storm_id: UUID
    count: int
    predictions: list[PredictionResponse]
