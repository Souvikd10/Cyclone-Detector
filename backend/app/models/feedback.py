"""
Feedback ORM model — Stage 17: Human-in-the-Loop forecaster actions.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Text, DateTime, ForeignKey, Enum as SAEnum, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.database import Base


class FeedbackAction(str, enum.Enum):
    """What the forecaster did with the prediction."""
    accept = "accept"
    override = "override"


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    storm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("storms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Denormalized for fast per-storm feedback lookups"
    )
    action: Mapped[FeedbackAction] = mapped_column(
        SAEnum(FeedbackAction, name="feedback_action"),
        nullable=False,
    )
    corrected_value: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Override data — the corrected prediction fields"
    )
    forecaster_note: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Free-text note from the forecaster"
    )
    forecaster_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="Identifier of the forecaster who submitted"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    prediction = relationship("Prediction", back_populates="feedbacks")
    storm = relationship("Storm", back_populates="feedbacks")

    __table_args__ = (
        Index("ix_feedback_storm_created", storm_id, created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<Feedback {self.action.value} on prediction {self.prediction_id}>"
