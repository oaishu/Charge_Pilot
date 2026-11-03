"""Forecast schemas — same contract as the backend (Section 4.5)."""
import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


Horizon = Literal["1h", "4h", "24h"]


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_kw: Optional[float] = None
    actual_kw: Optional[float] = None


class BaselineComparison(BaseModel):
    historical_average_kw: Optional[float] = None
    deviation_pct: Optional[float] = None


class ForecastData(BaseModel):
    time_series: List[ForecastPoint] = []
    baseline_comparison: Optional[BaselineComparison] = None


class ForecastListItem(BaseModel):
    id: uuid.UUID
    station_id: uuid.UUID
    station_name: Optional[str] = None
    forecast_horizon: str
    generated_at: datetime
    forecast_period_start: datetime
    forecast_period_end: datetime
    predicted_sessions: Optional[int] = None
    predicted_demand_kw: Optional[Decimal] = None
    confidence_level: Optional[Decimal] = None
    capacity_pressure: Optional[str] = None
    model_version: Optional[str] = None
    mae: Optional[Decimal] = None
    rmse: Optional[Decimal] = None
    is_simulated: bool

    model_config = ConfigDict(from_attributes=True)


class ForecastListResponse(BaseModel):
    data: List[ForecastListItem]


class ForecastDetail(ForecastListItem):
    forecast_data: Optional[ForecastData] = None


class ForecastGenerateRequest(BaseModel):
    station_id: uuid.UUID
    horizon: Horizon = "4h"


class ForecastGenerateResponse(BaseModel):
    id: uuid.UUID
    station_id: uuid.UUID
    forecast_horizon: str
    generated_at: datetime
    capacity_pressure: Optional[str] = None
    predicted_demand_kw: Optional[Decimal] = None
    message: str = "Forecast generated successfully"
