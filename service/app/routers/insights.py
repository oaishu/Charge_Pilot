"""Operational insight endpoints — same contract as the backend (operator role)."""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.errors import AppError
from app.core.security import require_role
from app.models import OperationalInsight, Station
from app.schemas.common import paginate_meta
from app.schemas.insight import (
    InsightListItem,
    InsightListResponse,
    InsightStatusUpdate,
    InsightUpdateResponse,
)

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("", response_model=InsightListResponse)
async def list_insights(
    station_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status", pattern="^(active|acknowledged|resolved)$"),
    priority: str | None = Query(None, pattern="^(low|medium|high)$"),
    insight_type: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(OperationalInsight)
    if station_id:
        stmt = stmt.where(OperationalInsight.station_id == station_id)
    if status_filter:
        stmt = stmt.where(OperationalInsight.status == status_filter)
    if priority:
        stmt = stmt.where(OperationalInsight.priority == priority)
    if insight_type:
        stmt = stmt.where(OperationalInsight.insight_type == insight_type)

    total = int(
        (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar() or 0
    )
    rows = (
        await db.execute(
            stmt.order_by(OperationalInsight.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    names: dict = {}
    if rows:
        names = {
            s.id: s.name
            for s in (
                await db.execute(
                    select(Station).where(Station.id.in_({r.station_id for r in rows if r.station_id}))
                )
            ).scalars()
        }

    data = [
        InsightListItem(
            id=i.id,
            station_id=i.station_id,
            station_name=names.get(i.station_id),
            insight_type=i.insight_type,
            title=i.title,
            summary=i.summary,
            suggested_action=i.suggested_action,
            status=i.status,
            priority=i.priority,
            source_type=i.source_type,
            source_id=i.source_id,
            created_at=i.created_at,
        )
        for i in rows
    ]
    return InsightListResponse(data=data, pagination=paginate_meta(total, page, page_size))


@router.patch("/{insight_id}", response_model=InsightUpdateResponse)
async def update_insight(
    insight_id: uuid.UUID,
    payload: InsightStatusUpdate,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    insight = (
        await db.execute(select(OperationalInsight).where(OperationalInsight.id == insight_id))
    ).scalar_one_or_none()
    if insight is None:
        raise AppError(
            "INSIGHT_NOT_FOUND",
            f"Insight with ID {insight_id} was not found",
            status_code=404,
        )

    now = datetime.now(timezone.utc)
    insight.status = payload.status
    if payload.status == "acknowledged":
        insight.acknowledged_at = insight.acknowledged_at or now
        insight.acknowledged_by = insight.acknowledged_by or getattr(operator, "username", "operator")
    insight.updated_at = now
    await db.commit()

    return InsightUpdateResponse(
        id=insight.id,
        status=insight.status,
        acknowledged_at=insight.acknowledged_at,
        acknowledged_by=insight.acknowledged_by,
        message="Insight acknowledged" if payload.status == "acknowledged" else "Insight updated",
        updated_at=now,
    )
