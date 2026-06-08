from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    TRAINING = "training"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class TriggerType(str, Enum):
    SCHEDULED = "scheduled"
    PERFORMANCE = "performance"
    DATA_DRIFT = "data_drift"


class RetrainStrategyConfig(BaseModel):
    model_name: str = ""
    trigger_type: TriggerType = TriggerType.SCHEDULED
    scheduled_interval_hours: int = 24
    performance_window_size: int = 10
    performance_f1_threshold: float = 0.7
    drift_kl_threshold: float = 0.5
    training_data_days: int = 30
    enabled: bool = True


class ModelVersionInfo(BaseModel):
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
    status: ModelStatus = ModelStatus.TRAINING
    parent_version_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


class TrainingProgress(BaseModel):
    model_id: str
    stage: str = "idle"
    current_step: int = 0
    total_steps: int = 4
    stage_description: str = ""
    data_size: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None


class TrainingContext(BaseModel):
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


class ABTestStatus(str, Enum):
    RUNNING = "running"
    COMPLETED_PROMOTED = "completed_promoted"
    COMPLETED_RETIRED = "completed_retired"


class ABTestConfig(BaseModel):
    model_name: str
    primary_model_id: str
    challenger_model_id: str
    primary_traffic_pct: float = 80.0
    min_windows: int = 5
    f1_improvement_threshold: float = 0.05
    status: ABTestStatus = ABTestStatus.RUNNING
    windows_completed: int = 0
    primary_precision: float = 0.0
    primary_recall: float = 0.0
    primary_f1: float = 0.0
    challenger_precision: float = 0.0
    challenger_recall: float = 0.0
    challenger_f1: float = 0.0
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")
    ended_at: str = ""


class ModelAlert(BaseModel):
    id: int = 0
    model_name: str
    model_id: str
    current_f1: float
    f1_threshold: float
    consecutive_low_windows: int
    suggestion: str
    dismissed: bool = False
    created_at: str = Field(default_factory=lambda: "")
    dismissed_at: str = ""


class ModelComparisonResult(BaseModel):
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
