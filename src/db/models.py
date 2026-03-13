"""SQLAlchemy ORM models for fire detection data."""
from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FireEvent(Base):
    """A single fire detection from FIRMS (VIIRS or MODIS)."""

    __tablename__ = "fire_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(10), nullable=False)  # 'VIIRS' | 'MODIS'
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[object] = mapped_column(
        Geography("POINT", srid=4326), nullable=False
    )
    frp: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str | None] = mapped_column(String(10))  # 'low'|'nominal'|'high'|numeric
    brightness: Mapped[float | None] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(50))

    alerts: Mapped[list["AlertLog"]] = relationship(back_populates="event")


class AreaOfInterest(Base):
    """A user-defined geographic area that can be monitored for fires."""

    __tablename__ = "areas_of_interest"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str | None] = mapped_column(String(100))
    email: Mapped[str | None] = mapped_column(String(255))
    boundary: Mapped[object] = mapped_column(
        Geography("POLYGON", srid=4326), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    alerts: Mapped[list["AlertLog"]] = relationship(back_populates="aoi")


class AlertLog(Base):
    """Record of an alert email sent for a fire event within an AOI."""

    __tablename__ = "alert_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aoi_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("areas_of_interest.id")
    )
    event_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("fire_events.id")
    )
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    aoi: Mapped["AreaOfInterest"] = relationship(back_populates="alerts")
    event: Mapped["FireEvent"] = relationship(back_populates="alerts")
