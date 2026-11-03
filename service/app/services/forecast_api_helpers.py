"""Shared helpers for forecast API serialization (ml/service port)."""
from typing import Dict

from app.models import DemandForecast
from app.schemas.forecast import ForecastListItem


def forecast_list_item(forecast: DemandForecast, station_names: Dict) -> ForecastListItem:
    station_name = station_names.get(forecast.station_id)
    return ForecastListItem(
        id=forecast.id,
        station_id=forecast.station_id,
        station_name=station_name,
        forecast_horizon=forecast.forecast_horizon,
        generated_at=forecast.generated_at,
        forecast_period_start=forecast.forecast_period_start,
        forecast_period_end=forecast.forecast_period_end,
        predicted_sessions=forecast.predicted_sessions,
        predicted_demand_kw=forecast.predicted_demand_kw,
        confidence_level=forecast.confidence_level,
        capacity_pressure=forecast.capacity_pressure,
        model_version=forecast.model_version,
        mae=forecast.mae,
        rmse=forecast.rmse,
        is_simulated=forecast.is_simulated,
    )
