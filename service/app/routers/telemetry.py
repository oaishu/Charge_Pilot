"""Telemetry ingestion + automatic fault detection (ML service's own feed).

This is the piece the backend wires through its simulation workflow: charge
points (or simulations) push ``telemetry_readings``, and the fault detector
scores each touched connector's recent window — trained GBDT first when an
artifact is staged, deterministic guardrails as the independent safety layer.
Readings above the detection threshold become ``fault_events``.
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.errors import AppError
from app.core.security import require_role
from app.models import Connector, TelemetryReading
from app.schemas.telemetry import (
    DetectedFaultBrief,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
)
from app.services.fault_detection_service import FaultDetectionService
from app.services.insight_service import OperationalInsightService

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("", response_model=TelemetryIngestResponse)
async def ingest_telemetry(
    payload: TelemetryIngestRequest,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Insert telemetry readings and (optionally) run fault detection over
    every connector that got new readings."""
    now = datetime.now(timezone.utc)
    connector_ids = {item.connector_id for item in payload.readings}
    connectors = {
        c.id: c
        for c in (
            await db.execute(select(Connector).where(Connector.id.in_(connector_ids)))
        ).scalars()
    }
    missing = connector_ids - set(connectors)
    if missing:
        raise AppError(
            "CONNECTOR_NOT_FOUND",
            f"Unknown connector(s): {', '.join(str(m) for m in sorted(map(str, missing)))}",
            status_code=404,
        )

    rows = [
        TelemetryReading(
            connector_id=item.connector_id,
            station_id=connectors[item.connector_id].station_id,
            recorded_at=item.recorded_at or now,
            power_kw=Decimal(str(item.power_kw)) if item.power_kw is not None else None,
            voltage_v=Decimal(str(item.voltage_v)) if item.voltage_v is not None else None,
            current_a=Decimal(str(item.current_a)) if item.current_a is not None else None,
            temperature_c=Decimal(str(item.temperature_c)) if item.temperature_c is not None else None,
            is_anomalous=item.is_anomalous,
            raw_payload=item.raw_payload,
            is_simulated=True,
        )
        for item in payload.readings
    ]
    db.add_all(rows)

    faults = []
    insight_ids = []
    if payload.run_detection:
        detector = FaultDetectionService()
        insight_service = OperationalInsightService(db)
        for connector_id in sorted(connectors):
            fault = await detector.detect(db, connectors[connector_id])
            if fault is not None:
                meta = (fault.evidence or {}).get("detector", {})
                faults.append(
                    DetectedFaultBrief(
                        id=fault.id,
                        connector_id=connector_id,
                        fault_type=fault.fault_type,
                        severity=fault.severity,
                        probability=meta.get("probability"),
                        detector_kind=meta.get("kind"),
                    )
                )
                # same workflow step as the backend: persisted fault → operator insight
                insight = await insight_service.generate_from_fault(
                    fault, connector=connectors[connector_id]
                )
                if insight is not None:
                    insight_ids.append(insight.id)

    await db.commit()
    return TelemetryIngestResponse(
        inserted=len(rows),
        connectors_scored=len(connectors) if payload.run_detection else 0,
        faults_detected=faults,
        insights_created=len(insight_ids),
        message=f"Ingested {len(rows)} readings; {len(faults)} fault(s), {len(insight_ids)} insight(s)",
    )
