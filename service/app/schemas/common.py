"""Shared schema primitives (pagination, error envelope) — same contract as the main backend."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


def paginate_meta(total: int, page: int, page_size: int) -> PaginationMeta:
    total_pages = max((total + page_size - 1) // page_size, 1) if total else 1
    return PaginationMeta(page=page, page_size=page_size, total=total, total_pages=total_pages)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class MessageResponse(BaseModel):
    message: str
    id: Optional[Any] = None
