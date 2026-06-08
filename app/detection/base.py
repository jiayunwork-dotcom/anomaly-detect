from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class AnomalyType(str, Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"


@dataclass
class AnomalyResult:
    timestamp: Any
    is_anomaly: bool
    score: float
    algorithm_name: str
    anomaly_type: AnomalyType


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        ...
