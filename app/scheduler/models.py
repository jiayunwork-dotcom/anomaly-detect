from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScheduleInterval(str, Enum):
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    CUSTOM = "custom"


class ScheduleStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"


class ScheduleCreateRequest(BaseModel):
    name: str = "Untitled Schedule"
    interval: ScheduleInterval = ScheduleInterval.MIN_5
    cron_expression: Optional[str] = None
    metrics: list[str] = []
    algorithms: list[str] = ["three_sigma", "iqr"]
    ensemble_mode: str = "majority"
    weights: dict[str, float] = {}


class ScheduleResponse(BaseModel):
    id: str
    name: str
    interval: str
    cron_expression: Optional[str] = None
    metrics: list[str]
    algorithms: list[str]
    ensemble_mode: str
    weights: dict[str, float]
    status: ScheduleStatus
    next_run_time: Optional[str] = None
    last_run_time: Optional[str] = None
    last_anomaly_count: int = 0
    last_alert_triggered: bool = False
    created_at: str = Field(default_factory=lambda: "")


class ExecutionHistoryResponse(BaseModel):
    id: int
    schedule_id: str
    start_time: str
    end_time: Optional[str] = None
    anomaly_count: int = 0
    alert_triggered: bool = False
    status: str = "pending"
    error: Optional[str] = None
