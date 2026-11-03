"""
Pydantic schemas for Storm API responses.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class StormBase(BaseModel):
    """Shared storm fields."""
    name: str = Field(..., description="Storm name, e.g. 'Cyclone Biparjoy'")
    basin: str = Field(default="NIO", description="Basin code")
    status: str = Field(..., description="active | historical | dissipated")
    category: Optional[str] = Field(None, description="IMD / Saffir-Simpson category")
    current_lat: float = Field(..., description="Current latitude")
    current_lon: float = Field(..., description="Current longitude")
    max_wind_speed: Optional[float] = Field(None, description="Max sustained wind (knots)")
    min_pressure: Optional[float] = Field(None, description="Min central pressure (hPa)")


class StormListItem(BaseModel):
    """Compact storm object for the list endpoint."""
    id: UUID
    name: str
    status: str
    category: Optional[str] = None
    current_lat: float
    current_lon: float
    max_wind_speed: Optional[float] = None
    min_pressure: Optional[float] = None
    last_updated: datetime

    model_config = {"from_attributes": True}


class StormDetail(StormBase):
    """Full storm detail."""
    id: UUID
    formed_at: datetime
    last_updated: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class StormListResponse(BaseModel):
    """Response wrapper for GET /storms."""
    count: int
    storms: list[StormListItem]
