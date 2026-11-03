"""Fault detection service — ml/service port of the backend's detector.

Same two-layer design (and same ordering guarantees) as
``backend/app/services/fault_detection_service.py``:

1. **deterministic guardrails** run first and stay the *independent safety
   layer* — overtemperature always wins, zero-power streaks and recurring
   anomalous readings are caught even when no artifact is staged;
2. the **trained GBDT** ( ``ml/artifacts/fault_model_v1.pkl`` ) re-scores the
   telemetry window and its result is used when it is at least as strong.
   Window semantics: the model scores the readings *since the most recent
   anomalous reading* (where a failure would start), so a fresh failure isn't
   drowned out by pre-fault healthy readings.

The scoring window comes from ``telemetry_readings`` written by the
``POST /telemetry`` ingestion endpoint (or passed explicitly).
"""
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import ml_artifacts
from app.core.config import settings
from app.models import Connector, FaultEvent, TelemetryReading

_LOGGER = logging.getLogger("ml.service.faults")

FAULT_TYPE_CONNECTOR_FAILURE = "connector_failure"
FAULT_TYPE_OVERTEMPERATURE = "overtemperature"
FAULT_TYPE_POWER_ANOMALY = "power_anomaly"
FAULT_TYPE_COMMUNICATION_LOSS = "communication_loss"
FAULT_TYPE_CUSTOM = "custom"


@dataclass
class DetectionResult:
    fault_type: str
    probability: float
    severity: str
    description: str
    suggested_action: str
    evidence: dict


class FaultDetectionService:
    """Trained-model detector with deterministic guardrails as safety net."""

    # Guardrail thresholds (independent safety layer, per plan Section 8.2)
    OVERTEMPERATURE_C = 85.0
    ZERO_POWER_STREAK_MINUTES = 10  # session active but no power for 10 minutes
    DETECTION_THRESHOLD = 0.60  # p(fault) threshold to persist a fault_event
    HISTORY_WINDOW_MINUTES = 30

    def classify(self, feature_window) -> DetectionResult:
        """Score a telemetry window. ``feature_window`` is a list of dicts with
        recorded_at / power_kw / temperature_c / is_anomalous keys."""
        now = datetime.now(timezone.utc)
        temps = [float(r.get("temperature_c") or 0) for r in feature_window]
        powers = [float(r.get("power_kw") or 0) for r in feature_window]
        anomalous = [bool(r.get("is_anomalous")) for r in feature_window]

        max_temp = max(temps) if temps else 0.0
        if max_temp >= self.OVERTEMPERATURE_C:
            return DetectionResult(
                fault_type=FAULT_TYPE_OVERTEMPERATURE,
                probability=0.95,
                severity="high" if max_temp >= 95 else "medium",
                description=f"Connector temperature reached {max_temp:.1f}°C, exceeding safe operating limits",
                suggested_action="Reduce charging power and inspect the connector cooling system",
                evidence={"max_temperature_c": max_temp, "threshold_c": self.OVERTEMPERATURE_C},
            )

        if powers and anomalous and sum(anomalous) >= len(anomalous) * 0.5:
            return DetectionResult(
                fault_type=FAULT_TYPE_POWER_ANOMALY,
                probability=0.75,
                severity="medium",
                description="Recurring anomalous telemetry readings detected in the recent window",
                suggested_action="Inspect connector hardware and check communication logs",
                evidence={"anomalous_fraction": sum(anomalous) / len(anomalous)},
            )

        # Zero power while session active for a sustained streak → connector failure.
        recent = [
            r
            for r in feature_window
            if (now - _as_utc(r["recorded_at"])) <= timedelta(minutes=self.ZERO_POWER_STREAK_MINUTES * 2)
        ]
        if recent and all(float(r.get("power_kw") or 0) == 0.0 for r in recent) and any(
            r.get("is_anomalous") for r in recent
        ):
            return DetectionResult(
                fault_type=FAULT_TYPE_CONNECTOR_FAILURE,
                probability=0.90,
                severity="high",
                description="Connector reported zero power output with anomalous readings during an active session",
                suggested_action="Inspect connector hardware and check communication logs",
                evidence={
                    "telemetry_readings": [
                        {
                            "recorded_at": _as_utc(r["recorded_at"]).isoformat(),
                            "power_kw": float(r.get("power_kw") or 0),
                            "is_anomalous": bool(r.get("is_anomalous")),
                        }
                        for r in recent[-5:]
                    ]
                },
            )

        return DetectionResult(
            fault_type="none",
            probability=0.0,
            severity="low",
            description="No fault indications in the telemetry window",
            suggested_action="",
            evidence={},
        )

    # ---- trained-model serving ----

    # artifact label → app fault type (the workflow surfaces)
    _ML_TYPE_MAP = {
        "no_energy_failure": FAULT_TYPE_CONNECTOR_FAILURE,  # failed charge → connector fault
        "power_anomaly": FAULT_TYPE_POWER_ANOMALY,
        "overlong_session": FAULT_TYPE_POWER_ANOMALY,  # stuck relay → anomalous power draw
    }

    _ML_DESCRIPTION = {
        FAULT_TYPE_CONNECTOR_FAILURE: "Trained detector classified this window as a failed charging session (no energy delivered)",
        FAULT_TYPE_POWER_ANOMALY: "Trained detector classified this window as abnormal power behavior",
    }

    _ML_ACTION = {
        FAULT_TYPE_CONNECTOR_FAILURE: "Inspect connector hardware and check session metering logs",
        FAULT_TYPE_POWER_ANOMALY: "Inspect connector power electronics and review telemetry history",
    }

    def _score_with_model(
        self, window: List[dict], connector: Connector
    ) -> Optional[DetectionResult]:
        """Score a telemetry window with the staged fault-detection artifact.

        Returns ``None`` when no artifact is staged, the window is empty, or
        the model classifies the window ``normal`` / below threshold (the
        bundle's ``decision_threshold`` is used so training and serving agree).
        """
        bundle = ml_artifacts.load_fault_artifact()
        if not bundle or not window:
            return None
        try:
            features = bundle["feature_names"]
            primary = bundle["primary"]
            threshold = float(bundle.get("decision_threshold", self.DETECTION_THRESHOLD))

            stamps = [datetime.fromisoformat(r["recorded_at"]) for r in window]
            recorded = [r.replace(tzinfo=timezone.utc) if r.tzinfo is None else r for r in stamps]
            span_hours = (max(recorded) - min(recorded)).total_seconds() / 3600.0
            powers = [float(r["power_kw"] or 0) for r in window]
            avg_kw = (sum(powers) / len(powers)) if powers else 0.0
            ts = max(recorded)
            is_fast = 1 if float(connector.max_power_kw or 0) >= 50 else 0

            values = {
                "duration_hours": max(span_hours, 0.25),  # window floor ≈ one observation step
                "avg_kw": avg_kw,
                "hour_start": ts.hour,
                "dow": ts.weekday(),
                "is_weekend": 1 if ts.weekday() >= 5 else 0,
                "is_fast": is_fast,
                "is_kr": 0,
            }
            row = [[float(values.get(name, 0.0)) for name in features]]

            proba = primary.predict_proba(row)[0]
            classes = list(primary.classes_)
            best_idx = max(
                (i for i, cls in enumerate(classes) if cls != "normal"),
                key=lambda i: proba[i],
                default=None,
            )
            if best_idx is None:
                return None
            label, probability = classes[best_idx], float(proba[best_idx])
            if label == "normal" or probability < threshold:
                return None

            fault_type = self._ML_TYPE_MAP.get(label, FAULT_TYPE_CUSTOM)
            severity = "high" if probability >= 0.9 else ("medium" if probability >= 0.75 else "low")
            return DetectionResult(
                fault_type=fault_type,
                probability=round(probability, 4),
                severity=severity,
                description=self._ML_DESCRIPTION.get(
                    fault_type, f"Trained detector class {label}"
                ),
                suggested_action=self._ML_ACTION.get(fault_type, "Inspect the connector"),
                evidence={
                    "detector": {
                        "kind": "ml",
                        "model_version": bundle.get("model_version", settings.ML_FAULT_MODEL_VERSION),
                        "label": label,
                        "class_probabilities": {
                            cls: round(float(p), 4) for cls, p in zip(classes, proba)
                        },
                        "window_features": {
                            k: round(v, 4) for k, v in values.items()
                        },
                    },
                },
            )
        except Exception:  # noqa: BLE001 - scoring must never break the workflow
            _LOGGER.exception("fault ML scoring failed - falling back to guardrails")
            return None

    async def detect(
        self,
        db: AsyncSession,
        connector: Connector,
        readings: Optional[List[TelemetryReading]] = None,
    ) -> Optional[FaultEvent]:
        """Build the feature window, score it (trained model first, guardrails
        as the independent safety layer), and persist a fault_event when the
        probability exceeds the detection threshold."""
        if readings is None:
            since = datetime.now(timezone.utc) - timedelta(minutes=self.HISTORY_WINDOW_MINUTES * 4)
            result = await db.execute(
                select(TelemetryReading)
                .where(
                    TelemetryReading.connector_id == connector.id,
                    TelemetryReading.recorded_at >= since,
                )
                .order_by(TelemetryReading.recorded_at.desc())
                .limit(50)
            )
            readings = list(result.scalars().all())

        window = [
            {
                "recorded_at": r.recorded_at,
                "power_kw": float(r.power_kw or 0),
                "temperature_c": float(r.temperature_c or 0),
                "is_anomalous": bool(r.is_anomalous),
            }
            for r in readings
        ]

        # 1) deterministic guardrails first — the independent safety layer
        #    (overtemperature always wins over scoring).
        detection = self.classify(window)

        # 2) trained-model scoring when an artifact is staged (and the safety
        #    layer hasn't already fired, or it fired weaker than the model).
        #    Readings are newest-first, so the scan stops at the newest anomaly.
        ml_window = list(window)
        for _i, _w in enumerate(window):
            if _w["is_anomalous"]:
                ml_window = window[: _i + 1]
                break
        ml_window = ml_window[:25]  # bound the slice (~2h at 5-min cadence)
        ml_result = self._score_with_model(
            [
                {
                    "recorded_at": r["recorded_at"].isoformat(),
                    "power_kw": r["power_kw"],
                    "temperature_c": r["temperature_c"],
                    "is_anomalous": r["is_anomalous"],
                }
                for r in ml_window
            ],
            connector,
        )
        if ml_result is not None and (
            detection.fault_type == "none" or ml_result.probability >= detection.probability
        ):
            detection = ml_result

        if detection.fault_type == "none" or detection.probability < self.DETECTION_THRESHOLD:
            return None

        # 3) connector-level dedupe: never stack a new fault on top of one
        #    the operator hasn't handled yet. A repeated anomaly burst for the
        #    same connector (continuous feed, retries, verify re-runs) keeps
        #    pointing at the existing open/investigating event instead of
        #    minting a fault (+insight) on every push.
        standing = await db.execute(
            select(FaultEvent.id)
            .where(
                FaultEvent.connector_id == connector.id,
                FaultEvent.status.in_(["open", "investigating"]),
            )
            .limit(1)
        )
        if standing.scalar_one_or_none() is not None:
            _LOGGER.info(
                "fault suppressed: connector %s already has an open/investigating fault",
                connector.id,
            )
            return None

        detector_meta = dict(detection.evidence.get("detector", {}))
        detector_meta.setdefault("probability", detection.probability)
        fault = FaultEvent(
            station_id=connector.station_id,
            connector_id=connector.id,
            fault_type=detection.fault_type,
            severity=detection.severity,
            status="open",
            detected_at=datetime.now(timezone.utc),
            evidence={
                # ML-dominant faults carry only detector metadata — attach the
                # newest raw readings so operators always see sample telemetry.
                "telemetry_readings": detection.evidence.get("telemetry_readings") or [
                    {
                        "recorded_at": _as_utc(w["recorded_at"]).isoformat(),
                        "power_kw": w["power_kw"],
                        "is_anomalous": w["is_anomalous"],
                    }
                    for w in window[:5]
                ],
                "connector_status_history": [
                    {"status": connector.status, "changed_at": connector.last_status_changed_at.isoformat()}
                ],
                "detector": {"model_version": settings.ML_FAULT_MODEL_VERSION, **detector_meta},
            },
            description=detection.description,
            suggested_action=detection.suggested_action,
            is_simulated=True,
        )
        db.add(fault)
        await db.flush()
        return fault


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


__all__ = ["FaultDetectionService", "DetectionResult"]
