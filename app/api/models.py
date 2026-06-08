from __future__ import annotations

from enum import Enum
from typing import Optional

from app.scheduler.models import (
    ExecutionHistoryResponse,
    ScheduleCreateRequest,
    ScheduleInterval,
    ScheduleResponse,
    ScheduleStatus,
)
from pydantic import BaseModel


class DataFormat(str, Enum):
    CSV = "csv"
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EnsembleMode(str, Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"


class LabelType(str, Enum):
    TP = "tp"
    FP = "fp"


class DetectionRequest(BaseModel):
    format: DataFormat
    data: str
    algorithms: list[str] = []
    algorithm_configs: dict[str, dict] = {}
    metric_configs: dict[str, dict] = {}
    ensemble_mode: EnsembleMode = EnsembleMode.MAJORITY
    weights: dict[str, float] = {}


class DetectionResponse(BaseModel):
    task_id: str
    status: TaskStatus


class AnomalyItem(BaseModel):
    timestamp: str
    metric: str
    is_anomaly: bool
    score: float
    algorithm: str
    anomaly_type: str


class DetectionResult(BaseModel):
    task_id: str
    status: TaskStatus
    anomalies: list[AnomalyItem] = []
    root_cause: Optional[dict] = None
    alerts: list[dict] = []


class BatchDetectionRequest(BaseModel):
    items: list[DetectionRequest]


class BatchDetectionResponse(BaseModel):
    task_ids: list[str]


class LabelRequest(BaseModel):
    event_id: int
    label: LabelType


class AnomalyEventResponse(BaseModel):
    id: int
    metric_name: str
    start_time: str
    end_time: str
    anomaly_type: str
    severity: float
    algorithm: str
    is_confirmed: bool
    confirmed_as: Optional[str] = None
    pattern_id: Optional[int] = None
    created_at: str


class DataImportRequest(BaseModel):
    format: DataFormat
    data: str


class DataImportResponse(BaseModel):
    metrics: list[str]
    rows_imported: int
