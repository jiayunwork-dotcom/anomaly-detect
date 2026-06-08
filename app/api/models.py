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


class ModelRegisterRequest(BaseModel):
    name: str
    algorithm_type: str
    training_params: dict = {}
    training_data_start: str = ""
    training_data_end: str = ""


class ModelVersionResponse(BaseModel):
    id: str
    name: str
    algorithm_type: str
    version: str
    training_params: dict = {}
    training_data_start: str = ""
    training_data_end: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    status: str = "training"
    parent_version_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class ModelGroupResponse(BaseModel):
    name: str
    version_count: int = 0
    latest_created: str = ""
    active_model_id: Optional[str] = None
    algorithm_type: str = ""
    active_f1: float = 0.0
    active_version: str = ""


class RetrainConfigRequest(BaseModel):
    model_name: str
    trigger_type: str = "scheduled"
    scheduled_interval_hours: int = 24
    performance_window_size: int = 10
    performance_f1_threshold: float = 0.7
    drift_kl_threshold: float = 0.5
    training_data_days: int = 30
    enabled: bool = True


class RetrainConfigResponse(BaseModel):
    model_name: str
    trigger_type: str = "scheduled"
    scheduled_interval_hours: int = 24
    performance_window_size: int = 10
    performance_f1_threshold: float = 0.7
    drift_kl_threshold: float = 0.5
    training_data_days: int = 30
    enabled: bool = True


class TrainingContextResponse(BaseModel):
    id: int
    model_id: str
    training_data_count: int = 0
    training_duration_seconds: float = 0.0
    stages: list[dict] = []
    old_precision: float = 0.0
    old_recall: float = 0.0
    old_f1: float = 0.0
    new_precision: float = 0.0
    new_recall: float = 0.0
    new_f1: float = 0.0
    auto_activated: bool = False
    error: Optional[str] = None
    completed_at: str = ""
    created_at: str = ""


class TrainingProgressResponse(BaseModel):
    model_id: str
    stage: str = "idle"
    current_step: int = 0
    total_steps: int = 4
    stage_description: str = ""
    data_size: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None


class ModelCompareRequest(BaseModel):
    model_a_id: str
    model_b_id: str


class ModelCompareResponse(BaseModel):
    model_a_id: str
    model_b_id: str
    model_a_version: str = ""
    model_b_version: str = ""
    model_a_scores: list[float] = []
    model_b_scores: list[float] = []
    model_a_precision: float = 0.0
    model_a_recall: float = 0.0
    model_a_f1: float = 0.0
    model_b_precision: float = 0.0
    model_b_recall: float = 0.0
    model_b_f1: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    ks_reject_null: bool = False
    model_a_mean: float = 0.0
    model_a_std: float = 0.0
    model_a_median: float = 0.0
    model_b_mean: float = 0.0
    model_b_std: float = 0.0
    model_b_median: float = 0.0
    sample_size_a: int = 0
    sample_size_b: int = 0
    sample_size: int = 0


class ABTestStartRequest(BaseModel):
    model_name: str
    primary_model_id: str
    challenger_model_id: str
    primary_traffic_pct: float = 80.0
    min_windows: int = 5
    f1_improvement_threshold: float = 0.05


class ABTestResponse(BaseModel):
    model_name: str
    primary_model_id: str
    challenger_model_id: str
    primary_traffic_pct: float = 80.0
    min_windows: int = 5
    f1_improvement_threshold: float = 0.05
    status: str = "running"
    windows_completed: int = 0
    primary_precision: float = 0.0
    primary_recall: float = 0.0
    primary_f1: float = 0.0
    challenger_precision: float = 0.0
    challenger_recall: float = 0.0
    challenger_f1: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    ended_at: str = ""


class ModelAlertResponse(BaseModel):
    id: int
    model_name: str
    model_id: str
    current_f1: float
    f1_threshold: float
    consecutive_low_windows: int
    suggestion: str
    dismissed: bool = False
    created_at: str = ""
    dismissed_at: str = ""
