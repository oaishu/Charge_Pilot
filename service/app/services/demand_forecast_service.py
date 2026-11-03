"""Demand forecast service (port of the backend's service for ml/service).

Same math, same serving semantics: a deterministic per-station
historical-average baseline, plus — when the trained artifact is staged and
beat the baseline — the RandomForest as a bounded, per-connector ratio
correction on top of it. ``model_version`` is stamped from the artifact.

Placement of imports differs from the backend (this package's own
``app.core``), everything else is behaviour-identical.
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import ml_artifacts
from app.core.config import settings
from app.models import ChargingSession, Connector, DemandForecast, Station

_LOGGER = logging.getLogger("ml.forecast")

HORIZON_HOURS: Dict[str, int] = {"1h": 1, "4h": 4, "24h": 24}
HORIZON_POINTS: Dict[str, int] = {"1h": 12, "4h": 16, "24h": 24}  # 5-min / 15-min / 1h steps


@dataclass
class ForecastPoint:
    timestamp: datetime
    predicted_kw: float
    predicted_sessions: float


@dataclass
class DemandForecastResult:
    station_id: uuid.UUID
    horizon: str
    forecast_period_start: datetime
    forecast_period_end: datetime
    predicted_sessions: int
    predicted_demand_kw: float
    confidence_level: float
    capacity_pressure: str
    baseline_average_kw: float
    deviation_pct: float
    points: List[ForecastPoint] = field(default_factory=list)
    model_version: str = settings.ML_FORECAST_MODEL_VERSION
    # trained-model serving metadata (None when the baseline path is used)
    ml_adjustment: Optional[dict] = None
    baseline_demand_kw: Optional[float] = None
    model_mae: Optional[float] = None
    model_rmse: Optional[float] = None


def _classify_pressure(predicted_kw: float, capacity_kw: float) -> str:
    if capacity_kw <= 0:
        return "high"
    ratio = predicted_kw / float(capacity_kw)
    if ratio < 0.60:
        return "low"
    if ratio < 0.85:
        return "medium"
    return "high"


def _hours_for(horizon: str) -> int:
    return HORIZON_HOURS.get(horizon, 4)


async def _load_hourly_profile(
    db: AsyncSession, station_id: uuid.UUID, ref: datetime, history_days: int
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int], int]:
    """Build (hour, is_weekend) -> avg simultaneous power kW profile from session history."""
    start = ref - timedelta(days=history_days)
    result = await db.execute(
        select(
            ChargingSession.started_at,
            ChargingSession.ended_at,
            ChargingSession.peak_power_kw,
        ).where(
            ChargingSession.station_id == station_id,
            ChargingSession.started_at >= start,
            ChargingSession.started_at < ref,
        )
    )
    rows = result.all()

    # Slot = (hour-of-day, is_weekend). Sum peak power of overlapping sessions per slot.
    slot_power: Dict[Tuple[int, int], float] = {}
    slot_sessions: Dict[Tuple[int, int], int] = {}
    for started_at, ended_at, peak_kw in rows:
        end = ended_at or (started_at + timedelta(hours=1))
        hour = started_at.hour
        weekend = 1 if started_at.weekday() >= 5 else 0
        slot_power[(hour, weekend)] = slot_power.get((hour, weekend), 0.0) + float(peak_kw or 0)
        slot_sessions[(hour, weekend)] = slot_sessions.get((hour, weekend), 0) + 1

    total_days = max(history_days, 1)
    profile: Dict[Tuple[int, int], float] = {}
    session_profile: Dict[Tuple[int, int], int] = {}
    for key in set(slot_power) | set(slot_sessions):
        # Aggregate load per hour slot, measured across all days - average concurrent power.
        profile[key] = slot_power.get(key, 0.0) / total_days
        session_profile[key] = int(round(slot_sessions.get(key, 0) / total_days))
    return profile, session_profile, len(rows)


async def _load_connector_profiles(
    db: AsyncSession, station_id: uuid.UUID, ref: datetime, history_days: int
) -> List[Tuple[Tuple[uuid.UUID, int], Dict[Tuple[int, int], float]]]:
    """Per-connector baseline profiles for trained-model serving.

    Returns ``[((connector_id, is_fast), profile), ...]`` where ``profile`` maps
    ``(hour, is_weekend)`` -> average slot power kW (same slot key as
    ``_load_hourly_profile``). Connectors without history are omitted - the ML
    correction never invents load without history.
    """
    start = ref - timedelta(days=history_days)
    result = await db.execute(
        select(
            ChargingSession.connector_id,
            ChargingSession.started_at,
            ChargingSession.ended_at,
            ChargingSession.peak_power_kw,
            Connector.max_power_kw,
        )
        .join(Connector, Connector.id == ChargingSession.connector_id)
        .where(
            ChargingSession.station_id == station_id,
            ChargingSession.started_at >= start,
            ChargingSession.started_at < ref,
        )
    )
    rows = result.all()

    profiles: Dict[uuid.UUID, Dict[Tuple[int, int], float]] = {}
    connector_type: Dict[uuid.UUID, bool] = {}
    totals = max(history_days, 1)
    for connector_id, started_at, _ended_at, peak_kw, max_kw in rows:
        hour = started_at.hour
        weekend = 1 if started_at.weekday() >= 5 else 0
        slot = (hour, weekend)
        profiles.setdefault(connector_id, {})
        profiles[connector_id][slot] = profiles[connector_id].get(slot, 0.0) + float(peak_kw or 0)
        connector_type[connector_id] = int(float(max_kw or 0)) >= 50

    out: List[Tuple[Tuple[uuid.UUID, int], Dict[Tuple[int, int], float]]] = []
    for connector_id, slot_power in profiles.items():
        profile = {slot: kw / totals for slot, kw in slot_power.items()}
        is_fast = connector_type.get(connector_id, False)
        out.append(((connector_id, int(is_fast)), profile))
    return out


class DemandForecastService:
    """Periodic-profile forecaster.

    Serves the deterministic statistical baseline, and - when a trained
    forecast artifact is staged (notebook 02's bundle) and its metrics beat
    that baseline - applies the RandomForest as a per-connector correction on
    top of it (``model_version`` stamped from the artifact).
    """

    HISTORY_DAYS = 90
    # ML correction guards: the artifact is per-charging-point kWh/h; the
    # serving value must never be invented where the baseline has no history.
    MIN_RATIO = 0.5
    MAX_RATIO = 1.5

    def __init__(self, model_version: Optional[str] = None):
        self.model_version = model_version or settings.ML_FORECAST_MODEL_VERSION

    def forecast_from_profile(
        self,
        profile: Dict[Tuple[int, int], float],
        session_profile: Dict[Tuple[int, int], int],
        start: datetime,
        horizon: str,
        capacity_kw: float,
    ) -> DemandForecastResult:
        hours = _hours_for(horizon)
        points_count = HORIZON_POINTS.get(horizon, hours)
        step_minutes = (hours * 60) // max(points_count, 1)

        points: List[ForecastPoint] = []
        total_kw = 0.0
        total_sessions = 0.0
        cur = start
        for _ in range(points_count):
            weekend = 1 if cur.weekday() >= 5 else 0
            kw = profile.get((cur.hour, weekend), 0.0)
            sessions = session_profile.get((cur.hour, weekend), 0)
            points.append(ForecastPoint(timestamp=cur, predicted_kw=round(kw, 2), predicted_sessions=sessions))
            total_kw += kw * (step_minutes / 60.0)
            total_sessions += sessions
            cur += timedelta(minutes=step_minutes)

        avg_kw = total_kw / hours if hours else 0.0
        forecast_end = start + timedelta(hours=hours)
        baseline_avg = (sum(profile.values()) / len(profile)) if profile else 0.0
        deviation = ((avg_kw - baseline_avg) / baseline_avg * 100.0) if baseline_avg > 0 else 0.0

        return DemandForecastResult(
            station_id=uuid.UUID(int=0),  # set by caller
            horizon=horizon,
            forecast_period_start=start,
            forecast_period_end=forecast_end,
            predicted_sessions=int(round(total_sessions)),
            predicted_demand_kw=round(total_kw, 2),
            confidence_level=0.75,
            capacity_pressure=_classify_pressure(avg_kw, capacity_kw),
            baseline_average_kw=round(baseline_avg, 2),
            deviation_pct=round(deviation, 1),
            points=points,
            model_version=self.model_version,
        )

    # ---- trained-model serving ----

    def _ml_point_row(
        self, ts: datetime, is_fast: int, hist_demand: float, features: List[str]
    ) -> List[float]:
        """Feature row for the hourly-demand model, in artifact feature order.

        ``hist_demand`` is this connector's baseline profile value - the same
        slot key the training-side feature used. Lag features are unavailable
        in the online path; the baseline value stands in for them (the training
        feature builder filled lags with the historical profile as fallback
        too, so this is distribution-matched).
        """
        ts = ts.replace(tzinfo=None) if ts.tzinfo is None else ts
        weekend = 1 if ts.weekday() >= 5 else 0
        values = {
            "hour": ts.hour,
            "dow": ts.weekday(),
            "is_weekend": weekend,
            "month": ts.month,
            "is_fast": is_fast,
            "is_kr": 0,
            "hist_demand": hist_demand,
            "lag_any": hist_demand,
            "roll3": hist_demand,
            "lag_recent": hist_demand,
        }
        return [float(values.get(name, 0.0)) for name in features]

    def _apply_ml_correction(
        self,
        result: DemandForecastResult,
        bundle: dict,
        connector_profiles: List[Tuple[Tuple[uuid.UUID, int], Dict[Tuple[int, int], float]]],
        start: datetime,
        horizon: str,
        capacity_kw: float,
    ) -> None:
        """Adjust the baseline forecast using the trained RandomForest.

        Per predictor step and per connector, the model re-scores the slot;
        the correction is a bounded ratio ``pred / baseline`` so the serving
        value never invents load where the baseline has none.
        """
        import numpy as np

        model = bundle["model_hourly_demand"]
        features = bundle["features_hourly"]
        hours = _hours_for(horizon)
        points_count = HORIZON_POINTS.get(horizon, hours)
        step_minutes = (hours * 60) // max(points_count, 1)

        ratios: List[float] = []
        for point in result.points:
            ts = point.timestamp
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            corrected = 0.0
            for (_conn_id, is_fast), profile in connector_profiles:
                if profile is None:
                    continue
                weekend = 1 if ts.weekday() >= 5 else 0
                hist = float(profile.get((ts.hour, weekend), 0.0))
                if hist <= 0.0:
                    continue  # no history for the slot - keep baseline (0)
                row = self._ml_point_row(ts, is_fast, hist, features)
                try:
                    pred = float(model.predict(np.array([row]))[0])
                except Exception:  # noqa: BLE001 - degrade to baseline per point
                    _LOGGER.warning("ML forecast scoring failed - keeping baseline", exc_info=True)
                    continue
                ratio = min(max(pred / hist, self.MIN_RATIO), self.MAX_RATIO)
                ratios.append(ratio)
                corrected += hist * ratio
            point.predicted_kw = round(corrected, 2)

        # station totals recomputed from corrected points
        total_kw = 0.0
        cur = start
        for point in result.points:
            total_kw += point.predicted_kw * (step_minutes / 60.0)
            cur += timedelta(minutes=step_minutes)
        result.predicted_demand_kw = round(total_kw, 2)
        avg_kw = total_kw / hours if hours else 0.0
        result.capacity_pressure = _classify_pressure(avg_kw, capacity_kw)
        result.model_version = bundle.get("model_version", self.model_version)
        result.confidence_level = 0.80
        result.ml_adjustment = {
            "model_version": bundle.get("model_version"),
            "connectors_scored": len(connector_profiles),
            "mean_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        }
        metrics = (bundle.get("metrics") or {}).get("model", {})
        m = metrics.get("hourly_demand", {})
        result.model_mae = m.get("mae")
        result.model_rmse = m.get("rmse")

    async def forecast(
        self, db: AsyncSession, station: Station, horizon: str = "4h"
    ) -> DemandForecastResult:
        """Generate a forecast and persist a ``demand_forecasts`` record."""
        now = datetime.now(timezone.utc)
        profile, session_profile, _ = await _load_hourly_profile(
            db, station.id, now, self.HISTORY_DAYS
        )
        result = self.forecast_from_profile(
            profile=profile,
            session_profile=session_profile,
            start=now.replace(minute=0, second=0, microsecond=0),
            horizon=horizon,
            capacity_kw=float(station.total_capacity_kw),
        )
        result.station_id = station.id

        # trained-model serving: only when the artifact beat the baseline
        bundle = ml_artifacts.load_forecast_artifact()
        if bundle is not None:
            connector_profiles = await _load_connector_profiles(
                db, station.id, now, self.HISTORY_DAYS
            )
            if connector_profiles:
                before_total = result.predicted_demand_kw
                self._apply_ml_correction(
                    result,
                    bundle,
                    connector_profiles,
                    result.forecast_period_start,
                    horizon,
                    capacity_kw=float(station.total_capacity_kw),
                )
                result.baseline_demand_kw = before_total

        record = DemandForecast(
            station_id=station.id,
            forecast_horizon=horizon,
            forecast_period_start=result.forecast_period_start,
            forecast_period_end=result.forecast_period_end,
            predicted_sessions=result.predicted_sessions,
            predicted_demand_kw=Decimal(result.predicted_demand_kw),
            confidence_level=Decimal(str(result.confidence_level)),
            capacity_pressure=result.capacity_pressure,
            model_version=result.model_version,
            mae=result.model_mae,  # artifact evaluation metrics (None on the baseline path)
            rmse=result.model_rmse,
            forecast_data={
                "time_series": [
                    {"timestamp": p.timestamp.isoformat(), "predicted_kw": p.predicted_kw, "actual_kw": None}
                    for p in result.points
                ],
                "baseline_comparison": {
                    "historical_average_kw": result.baseline_average_kw,
                    "deviation_pct": result.deviation_pct,
                },
                "ml_adjustment": result.ml_adjustment,
                "baseline_demand_kw": result.baseline_demand_kw,
            },
            is_simulated=True,
        )
        db.add(record)
        await db.flush()
        result.record_id = record.id  # type: ignore[attr-defined]
        return result


async def get_latest_forecast(db: AsyncSession, station_id: uuid.UUID) -> Optional[DemandForecast]:
    result = await db.execute(
        select(DemandForecast)
        .where(DemandForecast.station_id == station_id)
        .order_by(DemandForecast.generated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
