"""Minimal SQLAlchemy models for the standalone ML service.

These define this service's own schema, applied to its dedicated database by
``create_tables()`` (via ``scripts_setup_db.py``). Table/column names follow
the main project's conventions so moving data between the two systems stays
straightforward, but nothing is shared at runtime: no backend imports, no
shared migration state. Only the tables the ML endpoints need are mapped,
and no relationships are declared (all access by explicit join/select).
"""
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL, plain JSON elsewhere (matches the backend's portability trick)
JSONVariant = JSON().with_variant(JSONB(), "postgresql")
BigIntVariant = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), index=True)  # operator | driver
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Station(Base, TimestampMixin):
    __tablename__ = "stations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(100))
    location_area: Mapped[str] = mapped_column(String(100), index=True)
    latitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    longitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    total_capacity_kw: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    operator_notes: Mapped[Optional[str]] = mapped_column(Text)
    is_simulated: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(20), default="operational", index=True)


class Connector(Base, TimestampMixin):
    __tablename__ = "connectors"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    station_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), index=True
    )
    connector_type: Mapped[str] = mapped_column(String(30), index=True)
    max_power_kw: Mapped[Decimal] = mapped_column(Numeric(8, 2))
    status: Mapped[str] = mapped_column(String(20), default="available", index=True)
    last_status_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_simulated: Mapped[bool] = mapped_column(default=True)


class ChargingSession(Base):
    __tablename__ = "charging_sessions"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    connector_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("connectors.id", ondelete="CASCADE"), index=True
    )
    station_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    energy_kwh: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 3))
    peak_power_kw: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    is_simulated: Mapped[bool] = mapped_column(default=True)
    session_status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TelemetryReading(Base):
    """Charge-point telemetry (power/voltage/current/temperature), the raw input
    the fault detector scores. Same shape as the backend's table."""

    __tablename__ = "telemetry_readings"

    id: Mapped[int] = mapped_column(BigIntVariant, primary_key=True, autoincrement=True)
    connector_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("connectors.id", ondelete="CASCADE"), index=True
    )
    station_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    power_kw: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    voltage_v: Mapped[Optional[Decimal]] = mapped_column(Numeric(7, 2))
    current_a: Mapped[Optional[Decimal]] = mapped_column(Numeric(7, 2))
    temperature_c: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    is_anomalous: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSONVariant)
    is_simulated: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DemandForecast(Base, TimestampMixin):
    __tablename__ = "demand_forecasts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    station_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), index=True
    )
    forecast_horizon: Mapped[str] = mapped_column(String(20))  # 1h | 4h | 24h
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    forecast_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    forecast_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    predicted_sessions: Mapped[Optional[int]] = mapped_column(Integer)
    predicted_demand_kw: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    confidence_level: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    capacity_pressure: Mapped[Optional[str]] = mapped_column(String(20))
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    mae: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    rmse: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    forecast_data: Mapped[Optional[dict]] = mapped_column(JSONVariant)
    is_simulated: Mapped[bool] = mapped_column(default=True)


class FaultEvent(Base, TimestampMixin):
    __tablename__ = "fault_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    station_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), index=True
    )
    connector_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("connectors.id"), nullable=True
    )
    fault_type: Mapped[str] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(20), default="medium", index=True)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    evidence: Mapped[Optional[dict]] = mapped_column(JSONVariant)
    description: Mapped[Optional[str]] = mapped_column(Text)
    suggested_action: Mapped[Optional[str]] = mapped_column(Text)
    is_false_positive: Mapped[bool] = mapped_column(Boolean, default=False)
    is_simulated: Mapped[bool] = mapped_column(default=True)


class OperationalInsight(Base, TimestampMixin):
    __tablename__ = "operational_insights"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=new_uuid)
    station_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("stations.id", ondelete="CASCADE"), nullable=True, index=True
    )  # NULL = network-level insight
    insight_type: Mapped[str] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text)
    suggested_action: Mapped[Optional[str]] = mapped_column(Text)
    evidence: Mapped[Optional[dict]] = mapped_column(JSONVariant)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(100))
    source_type: Mapped[Optional[str]] = mapped_column(String(50))
    source_id: Mapped[Optional[uuid.UUID]] = mapped_column()
    priority: Mapped[str] = mapped_column(String(20), default="medium", index=True)
    is_simulated: Mapped[bool] = mapped_column(default=True)
