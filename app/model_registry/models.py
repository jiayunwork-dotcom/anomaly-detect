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
