"""Insight schemas — same contract as the backend (Section 4.7)."""
import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.common import PaginationMeta

InsightType = Literal["demand_spike", "capacity_pressure", "fault_detected", "recommendation"]
InsightStatus = Literal["active", "acknowledged", "resolved"]
InsightPriority = Literal["low", "medium", "high"]


class InsightListItem(BaseModel):
    id: uuid.UUID
    station_id: Optional[uuid.UUID] = None
    station_name: Optional[str] = None
    insight_type: str
    title: str
    summary: str
    suggested_action: Optional[str] = None
    status: str
    priority: str
    source_type: Optional[str] = None
    source_id: Optional[uuid.UUID] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InsightListResponse(BaseModel):
    data: List[InsightListItem]
    pagination: PaginationMeta


class InsightStatusUpdate(BaseModel):
    status: InsightStatus


class InsightUpdateResponse(BaseModel):
    id: uuid.UUID
    status: str
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    updated_at: datetime
    message: str
