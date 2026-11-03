"""
Pydantic schemas for CAP XML alert metadata (Stage 16).
"""

from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class AlertInfo(BaseModel):
    """Metadata about a generated CAP alert."""
    storm_id: UUID
    alert_id: str = Field(..., description="CAP alert identifier")
    status: str = Field(default="Test", description="CAP status: Test (always for prototype)")
    msg_type: str = Field(default="Alert", description="Alert | Update | Cancel")
    severity: str = Field(..., description="Minor | Moderate | Severe | Extreme")
    urgency: str = Field(..., description="Immediate | Expected | Future | Past")
    certainty: str = Field(..., description="Observed | Likely | Possible | Unlikely")
    headline: str
    description: str
    effective: datetime
    expires: Optional[datetime] = None
    sender: str = Field(default="SIH-2K26 Cyclone Prediction System (PROTOTYPE)")


class AlertResponse(BaseModel):
    """Response wrapper for GET /alerts/{storm_id}."""
    info: AlertInfo
    cap_xml: str = Field(..., description="Full CAP 1.2 XML document as string")
