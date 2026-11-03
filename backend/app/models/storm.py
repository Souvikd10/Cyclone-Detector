"""
Storm ORM model — represents a tropical cyclone being tracked.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Float, DateTime, Enum as SAEnum, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry
import enum

from app.database import Base


class StormStatus(str, enum.Enum):
    """Lifecycle status of a tracked storm."""
    active = "active"
    historical = "historical"
    dissipated = "dissipated"


class Storm(Base):
    __tablename__ = "storms"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    basin: Mapped[str] = mapped_column(
        String(50), nullable=False, default="NIO",
        comment="Basin code, e.g. NIO (North Indian Ocean)"
    )
    status: Mapped[StormStatus] = mapped_column(
        SAEnum(StormStatus, name="storm_status"),
        nullable=False,
        default=StormStatus.active,
        index=True,
    )
    category: Mapped[str | None] = mapped_column(
        String(30), nullable=True,
        comment="IMD / Saffir-Simpson category"
    )
    current_lat: Mapped[float] = mapped_column(Float, nullable=False)
    current_lon: Mapped[float] = mapped_column(Float, nullable=False)

    # PostGIS geometry column for spatial queries
    geom = mapped_column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=True,
        comment="Current position as PostGIS point"
    )

    max_wind_speed: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Max sustained wind in knots"
    )
    min_pressure: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Minimum central pressure in hPa"
    )

    formed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    predictions = relationship("Prediction", back_populates="storm", lazy="selectin")
    feedbacks = relationship("Feedback", back_populates="storm", lazy="selectin")

    # Indexes
    __table_args__ = (
        Index("ix_storms_geom", geom, postgresql_using="gist"),
        Index("ix_storms_status_updated", status, last_updated.desc()),
    )

    def __repr__(self) -> str:
        return f"<Storm {self.name} ({self.status.value})>"
