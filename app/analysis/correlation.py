from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from dtw import dtw
from scipy.stats import pearsonr


@dataclass
class RelatedPair:
    metric_a: str
    metric_b: str
    score: float
    method: str


@dataclass
class CorrelationResult:
    pearson_matrix: pd.DataFrame
    dtw_matrix: pd.DataFrame
    related_pairs: list[RelatedPair]


def _compute_pearson_matrix(data: pd.DataFrame) -> pd.DataFrame:
    metrics = data.columns.tolist()
    n = len(metrics)
    matrix = pd.DataFrame(np.ones((n, n)), index=metrics, columns=metrics)
    for i, j in combinations(range(n), 2):
        a = data.iloc[:, i].dropna()
        b = data.iloc[:, j].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 3:
            matrix.iloc[i, j] = 0.0
            matrix.iloc[j, i] = 0.0
            continue
        r, _ = pearsonr(a.loc[common].values, b.loc[common].values)
        matrix.iloc[i, j] = r
        matrix.iloc[j, i] = r
    return matrix


def _compute_dtw_matrix(
    data: pd.DataFrame, max_lag_window: int = 60
) -> pd.DataFrame:
    metrics = data.columns.tolist()
    n = len(metrics)
    matrix = pd.DataFrame(np.zeros((n, n)), index=metrics, columns=metrics)
    for i, j in combinations(range(n), 2):
        a = data.iloc[:, i].dropna().values
        b = data.iloc[:, j].dropna().values
        if len(a) == 0 or len(b) == 0:
            matrix.iloc[i, j] = np.inf
            matrix.iloc[j, i] = np.inf
            continue
        alignment = dtw(
            a, b, step_pattern="symmetric2", open_end=False, open_begin=False
        )
        distance = alignment.distance
        norm = np.sqrt(len(a) * len(b))
        normalized = distance / norm if norm > 0 else np.inf
        matrix.iloc[i, j] = normalized
        matrix.iloc[j, i] = normalized
    return matrix


def run_correlation_analysis(
    data: pd.DataFrame,
    anomaly_window: tuple[int, int],
    pearson_threshold: float = 0.7,
    dtw_threshold: float = 1.0,
    max_lag_window: int = 60,
) -> CorrelationResult:
    start, end = anomaly_window
    window_data = data.iloc[start:end]
    pearson_matrix = _compute_pearson_matrix(window_data)
    dtw_matrix = _compute_dtw_matrix(window_data, max_lag_window)
    related_pairs: list[RelatedPair] = []
    metrics = window_data.columns.tolist()
    for i, j in combinations(range(len(metrics)), 2):
        r = pearson_matrix.iloc[i, j]
        if abs(r) > pearson_threshold:
            related_pairs.append(
                RelatedPair(
                    metric_a=metrics[i],
                    metric_b=metrics[j],
                    score=float(r),
                    method="pearson",
                )
            )
        d = dtw_matrix.iloc[i, j]
        if d < dtw_threshold:
            related_pairs.append(
                RelatedPair(
                    metric_a=metrics[i],
                    metric_b=metrics[j],
                    score=float(d),
                    method="dtw",
                )
            )
    return CorrelationResult(
        pearson_matrix=pearson_matrix,
        dtw_matrix=dtw_matrix,
        related_pairs=related_pairs,
    )
