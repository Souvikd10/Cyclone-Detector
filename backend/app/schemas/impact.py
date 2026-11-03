"""
Pydantic schemas for the Impact endpoint (Stage 14).
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Any, Optional


class LandfallLocation(BaseModel):
    """Estimated landfall point."""
    lat: float = Field(..., description="Latitude of landfall point")
    lon: float = Field(..., description="Longitude of landfall point")
    name: Optional[str] = Field(None, description="Nearest named location")
    country: Optional[str] = Field(None, description="Country code")
    state: Optional[str] = Field(None, description="State or province")
    distance_km: Optional[float] = Field(None, description="Distance from current position to landfall")


class ImpactZone(BaseModel):
    """Potential coastal impact area as GeoJSON."""
    type: str = "Feature"
    geometry: dict[str, Any] = Field(
        ..., description="GeoJSON Polygon/MultiPolygon of the impact zone"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Impact zone metadata (wind category, surge risk, etc.)"
    )


class ImpactResponse(BaseModel):
    """Full Stage 14 response for GET /storms/{id}/impact."""
    storm_id: UUID
    prediction_id: UUID
    landfall_location: Optional[LandfallLocation] = None
    landfall_eta: Optional[datetime] = None
    impact_zone: Optional[ImpactZone] = None
    wind_radius_km: Optional[float] = Field(None, description="Estimated destructive wind radius")
    storm_surge_risk: Optional[str] = Field(None, description="low | moderate | high | extreme")
    population_exposed: Optional[int] = Field(None, description="Estimated population in impact zone")
    computed_at: datetime

    model_config = {"from_attributes": True}
