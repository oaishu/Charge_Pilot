"""Fault event endpoints — same contract as the backend (operator role).

Faults are *written* by the detector (see ``POST /telemetry`` and the seeded
history); these endpoints list/inspect/update them.
"""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.errors import AppError
from app.core.security import require_role
from app.models import Connector, FaultEvent, Station
from app.schemas.common import paginate_meta
from app.schemas.fault import (
    FaultDetail,
    FaultEvidence,
    FaultListItem,
    FaultListResponse,
    FaultStatusUpdate,
    FaultUpdateResponse,
)

router = APIRouter(prefix="/faults", tags=["faults"])


@router.get("", response_model=FaultListResponse)
async def list_faults(
    station_id: uuid.UUID | None = None,
    status: str | None = Query(None, pattern="^(open|investigating|resolved|false_positive)$"),
    severity: str | None = Query(None, pattern="^(low|medium|high|critical)$"),
    fault_type: str | None = None,
    from_: datetime | None = Query(None, alias="from"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(FaultEvent)
    if station_id:
        stmt = stmt.where(FaultEvent.station_id == station_id)
    if status:
        stmt = stmt.where(FaultEvent.status == status)
    if severity:
        stmt = stmt.where(FaultEvent.severity == severity)
    if fault_type:
        stmt = stmt.where(FaultEvent.fault_type == fault_type)
    if from_:
        stmt = stmt.where(FaultEvent.detected_at >= from_)

    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar() or 0)
    rows = (
        await db.execute(
            stmt.order_by(FaultEvent.detected_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    station_names, connector_types = {}, {}
    if rows:
        station_names = {
            s.id: s.name
            for s in (await db.execute(select(Station).where(Station.id.in_({r.station_id for r in rows})))).scalars()
        }
        cids = {r.connector_id for r in rows if r.connector_id}
        if cids:
            connector_types = {
                c.id: c.connector_type
                for c in (await db.execute(select(Connector).where(Connector.id.in_(cids)))).scalars()
            }

    data = [
        FaultListItem(
            id=f.id,
            station_id=f.station_id,
            station_name=station_names.get(f.station_id),
            connector_id=f.connector_id,
            connector_type=connector_types.get(f.connector_id),
            fault_type=f.fault_type,
            severity=f.severity,
            status=f.status,
            detected_at=f.detected_at,
            resolved_at=f.resolved_at,
            description=f.description,
            suggested_action=f.suggested_action,
            is_false_positive=f.is_false_positive,
            is_simulated=f.is_simulated,
        )
        for f in rows
    ]
    return FaultListResponse(data=data, pagination=paginate_meta(total, page, page_size))


@router.get("/{fault_id}", response_model=FaultDetail)
async def get_fault(
    fault_id: uuid.UUID,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    fault = (
        await db.execute(select(FaultEvent).where(FaultEvent.id == fault_id))
    ).scalar_one_or_none()
    if fault is None:
        raise AppError(
            "FAULT_NOT_FOUND",
            f"Fault with ID {fault_id} was not found",
            status_code=404,
        )
    station = (
        await db.execute(select(Station).where(Station.id == fault.station_id))
    ).scalar_one_or_none()
    connector = None
    if fault.connector_id:
        connector = (
            await db.execute(select(Connector).where(Connector.id == fault.connector_id))
        ).scalar_one_or_none()

    evidence = FaultEvidence.model_validate(fault.evidence or {})
    return FaultDetail(
        id=fault.id,
        station_id=fault.station_id,
        station_name=station.name if station else "",
        connector_id=fault.connector_id,
        connector_type=connector.connector_type if connector else None,
        fault_type=fault.fault_type,
        severity=fault.severity,
        status=fault.status,
        detected_at=fault.detected_at,
        resolved_at=fault.resolved_at,
        description=fault.description,
        suggested_action=fault.suggested_action,
        evidence=evidence,
        is_false_positive=fault.is_false_positive,
        is_simulated=fault.is_simulated,
    )


@router.patch("/{fault_id}", response_model=FaultUpdateResponse)
async def update_fault(
    fault_id: uuid.UUID,
    payload: FaultStatusUpdate,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    fault = (
        await db.execute(select(FaultEvent).where(FaultEvent.id == fault_id))
    ).scalar_one_or_none()
    if fault is None:
        raise AppError(
            "FAULT_NOT_FOUND",
            f"Fault with ID {fault_id} was not found",
            status_code=404,
        )

    fault.status = payload.status
    if payload.is_false_positive is not None:
        fault.is_false_positive = payload.is_false_positive
    if payload.status == "resolved":
        fault.resolved_at = datetime.now(timezone.utc)
    if payload.notes:
        fault.evidence = {**(fault.evidence or {}), "operator_notes": payload.notes}
    fault.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return FaultUpdateResponse(id=fault.id, status=fault.status, updated_at=fault.updated_at)
