"""Fault schemas — same contract as the backend (Section 4.6)."""
import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.common import PaginationMeta

FaultStatus = Literal["open", "investigating", "resolved", "false_positive"]
FaultType = Literal[
    "connector_failure", "overtemperature", "power_anomaly", "communication_loss", "custom"
]
Severity = Literal["low", "medium", "high", "critical"]


class TelemetryEvidence(BaseModel):
    recorded_at: datetime
    power_kw: Optional[float] = None
    is_anomalous: Optional[bool] = None


class StatusHistoryEntry(BaseModel):
    status: str
    changed_at: datetime


class FaultEvidence(BaseModel):
    telemetry_readings: List[TelemetryEvidence] = []
    connector_status_history: List[StatusHistoryEntry] = []


class FaultListItem(BaseModel):
    id: uuid.UUID
    station_id: uuid.UUID
    station_name: Optional[str] = None
    connector_id: Optional[uuid.UUID] = None
    connector_type: Optional[str] = None
    fault_type: str
    severity: str
    status: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    description: Optional[str] = None
    suggested_action: Optional[str] = None
    is_false_positive: bool
    is_simulated: bool

    model_config = ConfigDict(from_attributes=True)


class FaultListResponse(BaseModel):
    data: List[FaultListItem]
    pagination: PaginationMeta


class FaultDetail(FaultListItem):
    evidence: Optional[FaultEvidence] = None


class FaultStatusUpdate(BaseModel):
    status: FaultStatus
    is_false_positive: Optional[bool] = None
    notes: Optional[str] = None


class FaultUpdateResponse(BaseModel):
    id: uuid.UUID
    status: str
    updated_at: datetime
    message: str = "Fault status updated successfully"
