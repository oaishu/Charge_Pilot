"""Operational insight service (port of the backend's service for ml/service).

Generates operator insights from forecasts (capacity pressure) and detected
faults, with plain deterministic action text. Insights live in this service's
own database.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Connector, DemandForecast, FaultEvent, OperationalInsight, Station

SUGGESTED_ACTIONS = {
    "capacity_pressure": "Consider redirecting high-demand sessions to nearby stations or reducing session time limits.",
    "fault_detected": "Inspect the affected connector and check communication logs.",
    "demand_spike": "Monitor network load and prepare for increased demand.",
}


def get_suggested_action(insight_type: str, context: dict) -> str:
    """Deterministic suggested-action mapping (text templating only)."""
    if insight_type in SUGGESTED_ACTIONS:
        return SUGGESTED_ACTIONS[insight_type]
    return "Review current network conditions and update action if needed."


class OperationalInsightService:
    """Generates operator insights from forecasts and detected faults."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_from_forecast(
        self, forecast: DemandForecast, station: Optional[Station] = None
    ) -> Optional[OperationalInsight]:
        """Create an insight when capacity pressure is high (or medium)."""
        if forecast.capacity_pressure not in ("high", "medium"):
            return None

        if station is None:
            result = await self.db.execute(select(Station).where(Station.id == forecast.station_id))
            station = result.scalar_one_or_none()
        if station is None:
            return None

        # Avoid duplicating an already-active insight for the same source.
        existing = await self.db.execute(
            select(OperationalInsight).where(
                OperationalInsight.source_type == "forecast",
                OperationalInsight.source_id == forecast.id,
                OperationalInsight.status == "active",
            )
        )
        if existing.scalar_one_or_none() is not None:
            return None

        pressure = forecast.capacity_pressure
        predicted_kw = float(forecast.predicted_demand_kw or 0)
        capacity = float(station.total_capacity_kw)
        pct = (predicted_kw / capacity * 100.0) if capacity else 0.0
        priority = "high" if pressure == "high" else "medium"

        insight = OperationalInsight(
            station_id=station.id,
            insight_type="capacity_pressure",
            title=f"{'High' if pressure == 'high' else 'Elevated'} demand predicted at {station.name}",
            summary=(
                f"Demand forecast indicates {predicted_kw:.1f} kW load over the next "
                f"{forecast.forecast_horizon}, exceeding {pct:.1f}% of rated capacity."
            ),
            suggested_action=get_suggested_action("capacity_pressure", {}),
            evidence={
                "forecast_id": str(forecast.id),
                "predicted_demand_kw": predicted_kw,
                "capacity_kw": capacity,
                "capacity_usage_pct": round(pct, 1),
            },
            status="active",
            source_type="forecast",
            source_id=forecast.id,
            priority=priority,
            is_simulated=True,
        )
        self.db.add(insight)
        await self.db.flush()
        return insight

    async def generate_from_fault(
        self,
        fault: FaultEvent,
        station: Optional[Station] = None,
        connector: Optional[Connector] = None,
    ) -> Optional[OperationalInsight]:
        """Create an insight with suggested action and evidence link."""
        if station is None:
            result = await self.db.execute(select(Station).where(Station.id == fault.station_id))
            station = result.scalar_one_or_none()
        if station is None:
            return None

        existing = await self.db.execute(
            select(OperationalInsight).where(
                OperationalInsight.source_type == "fault_detection",
                OperationalInsight.source_id == fault.id,
                OperationalInsight.status == "active",
            )
        )
        if existing.scalar_one_or_none() is not None:
            return None

        severity = fault.severity
        priority = {"critical": "high", "high": "high", "medium": "medium", "low": "low"}.get(severity, "medium")

        connector_label = ""
        if connector is None and fault.connector_id:
            c_result = await self.db.execute(select(Connector).where(Connector.id == fault.connector_id))
            connector = c_result.scalar_one_or_none()
        if connector:
            connector_label = f" ({connector.connector_type})"

        insight = OperationalInsight(
            station_id=station.id,
            insight_type="fault_detected",
            title=f"{fault.fault_type.replace('_', ' ').title()} detected at {station.name}{connector_label}",
            summary=fault.description or f"A {fault.fault_type} fault was detected with {severity} severity.",
            suggested_action=fault.suggested_action or get_suggested_action("fault_detected", {}),
            evidence={
                "fault_id": str(fault.id),
                "connector_id": str(fault.connector_id) if fault.connector_id else None,
                "severity": severity,
            },
            status="active",
            source_type="fault_detection",
            source_id=fault.id,
            priority=priority,
            is_simulated=True,
        )
        self.db.add(insight)
        await self.db.flush()
        return insight


async def ack_or_resolve(
    db: AsyncSession, insight: OperationalInsight, status: str, acknowledged_by: Optional[str]
) -> OperationalInsight:
    insight.status = status
    if status == "acknowledged" or (status == "resolved" and not insight.acknowledged_at):
        insight.acknowledged_at = insight.acknowledged_at or datetime.now(timezone.utc)
        insight.acknowledged_by = acknowledged_by or insight.acknowledged_by or "operator"
    await db.flush()
    return insight
