"""
Prediction ORM model — stores Member A's AI predictions + Stage 14 impact data.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    storm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("storms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # ── Member A's prediction sub-objects (stored as JSONB) ──
    detection: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    structure: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    current_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    forecast: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment='{"track": [...], "intensity": [...]}'
    )
    ri: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Rapid intensification data"
    )
    uncertainty: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Uncertainty cone / ensemble spread"
    )
    xai: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Explainability / attribution data"
    )

    # ── Stage 14: Impact data (computed by this backend) ──
    impact: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment='{"landfall_location": {...}, "landfall_eta": "...", "impact_zone_geojson": {...}}'
    )

    # ── MLflow link ──
    model_run_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="MLflow run ID that produced this prediction"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    storm = relationship("Storm", back_populates="predictions")
    feedbacks = relationship("Feedback", back_populates="prediction", lazy="selectin")

    # Indexes — BRIN on timestamp for efficient time-range queries
    __table_args__ = (
        Index(
            "ix_predictions_storm_timestamp",
            storm_id, timestamp.desc()
        ),
    )

    def __repr__(self) -> str:
        return f"<Prediction {self.id} for storm {self.storm_id}>"
