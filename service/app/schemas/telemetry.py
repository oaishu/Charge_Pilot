"""Telemetry ingestion schemas — the ML service's own charge-point feed."""
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TelemetryInItem(BaseModel):
    connector_id: uuid.UUID
    recorded_at: Optional[datetime] = None          # defaults to ingestion time
    power_kw: Optional[float] = None
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    temperature_c: Optional[float] = None
    is_anomalous: bool = False
    raw_payload: Optional[dict] = None


class TelemetryIngestRequest(BaseModel):
    readings: List[TelemetryInItem] = Field(min_length=1, max_length=500)
    run_detection: bool = True  # score each touched connector's window after insert


class DetectedFaultBrief(BaseModel):
    id: uuid.UUID
    connector_id: uuid.UUID
    fault_type: str
    severity: str
    probability: Optional[float] = None
    detector_kind: Optional[str] = None  # "ml" | "guardrail"


class TelemetryIngestResponse(BaseModel):
    inserted: int
    connectors_scored: int
    faults_detected: List[DetectedFaultBrief]
    insights_created: int = 0
    message: str
