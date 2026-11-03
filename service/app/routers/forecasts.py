"""Forecast endpoints for the standalone ML service — same contract as the backend."""
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.errors import AppError
from app.core.security import require_role
from app.models import DemandForecast, Station
from app.schemas.forecast import (
    ForecastData,
    ForecastDetail,
    ForecastGenerateRequest,
    ForecastGenerateResponse,
    ForecastListResponse,
)
from app.services.demand_forecast_service import DemandForecastService
from app.services.forecast_api_helpers import forecast_list_item
from app.services.insight_service import OperationalInsightService

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("", response_model=ForecastListResponse)
async def list_forecasts(
    station_id: uuid.UUID | None = None,
    horizon: str | None = Query(None, pattern="^(1h|4h|24h)$"),
    capacity_pressure: str | None = Query(None, pattern="^(low|medium|high)$"),
    from_: datetime | None = Query(None, alias="from"),
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(DemandForecast)
    if station_id:
        stmt = stmt.where(DemandForecast.station_id == station_id)
    if horizon:
        stmt = stmt.where(DemandForecast.forecast_horizon == horizon)
    if capacity_pressure:
        stmt = stmt.where(DemandForecast.capacity_pressure == capacity_pressure)
    if from_:
        stmt = stmt.where(DemandForecast.generated_at >= from_)
    rows = (
        await db.execute(stmt.order_by(DemandForecast.generated_at.desc()).limit(200))
    ).scalars().all()

    names: dict = {}
    if rows:
        names = {
            s.id: s.name
            for s in (
                await db.execute(select(Station).where(Station.id.in_({r.station_id for r in rows})))
            ).scalars()
        }
    return ForecastListResponse(data=[forecast_list_item(r, names) for r in rows])


@router.get("/{forecast_id}", response_model=ForecastDetail)
async def get_forecast(
    forecast_id: uuid.UUID,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    forecast = (
        await db.execute(select(DemandForecast).where(DemandForecast.id == forecast_id))
    ).scalar_one_or_none()
    if forecast is None:
        raise AppError(
            "FORECAST_NOT_FOUND",
            f"Forecast with ID {forecast_id} was not found",
            status_code=404,
        )
    station = (
        await db.execute(select(Station).where(Station.id == forecast.station_id))
    ).scalar_one_or_none()
    item = forecast_list_item(forecast, {forecast.station_id: station.name if station else ""})

    data = forecast.forecast_data or {}
    return ForecastDetail(
        **item.model_dump(),
        forecast_data=ForecastData.model_validate(data) if data else None,
    )


@router.post("/generate", response_model=ForecastGenerateResponse, status_code=201)
async def generate_forecast(
    payload: ForecastGenerateRequest,
    operator=Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    station = (
        await db.execute(select(Station).where(Station.id == payload.station_id))
    ).scalar_one_or_none()
    if station is None:
        raise AppError(
            "STATION_NOT_FOUND",
            f"Station with ID {payload.station_id} was not found",
            status_code=404,
        )
    result = await DemandForecastService().forecast(db, station, payload.horizon)
    record_id = getattr(result, "record_id", None)

    # same workflow step as the backend: a forecast under pressure raises an
    # operator insight (deduped per source forecast by the insight service)
    if record_id is not None:
        record = await db.get(DemandForecast, record_id)
        if record is not None:
            await OperationalInsightService(db).generate_from_forecast(record, station)

    await db.commit()
    return ForecastGenerateResponse(
        id=record_id,
        station_id=station.id,
        forecast_horizon=payload.horizon,
        generated_at=datetime.now(),
        capacity_pressure=result.capacity_pressure,
        predicted_demand_kw=result.predicted_demand_kw,
        message="Forecast generated successfully",
    )
