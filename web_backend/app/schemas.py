from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiErrorBody(BaseModel):
    code: str
    message: str
    hint: str | None = None
    details: dict[str, Any] | None = None


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    api: str = "gui2-fastapi"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VersionResponse(BaseModel):
    api: str = "v1"
    backend: str = "fastapi"
    cli: str = "unknown"
    runner: str = "unknown"


class JobAccepted(BaseModel):
    job_id: str
    state: Literal["pending", "running", "ok", "error", "cancelled"] = "pending"


class PhaseEvent(BaseModel):
    type: Literal[
        "phase_start",
        "phase_progress",
        "phase_end",
        "run_end",
        "queue_progress",
        "log_line",
    ]
    run_id: str
    filter: str | None = None
    phase: str | None = None
    pct: float | None = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
