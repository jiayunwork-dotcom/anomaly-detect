from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests


@dataclass
class CausalEdge:
    cause: str
    effect: str
    p_value: float
    f_statistic: float


def _granger_test_pair(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> tuple[float, float]:
    data = np.column_stack([y, x])
    try:
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        best_p = 1.0
        best_f = 0.0
        for lag in range(1, max_lag + 1):
            ssr_ftest = result[lag][0]["ssr_ftest"]
            p = ssr_ftest[1]
            f = ssr_ftest[0]
            if p < best_p:
                best_p = p
                best_f = f
        return best_p, best_f
    except Exception:
        return 1.0, 0.0


def run_granger_causality(
    data: pd.DataFrame,
    anomaly_window: tuple[int, int],
    max_lag: int = 5,
    significance: float = 0.05,
) -> list[CausalEdge]:
    start, end = anomaly_window
    window_data = data.iloc[start:end]
    metrics = window_data.columns.tolist()
    raw_edges: list[CausalEdge] = []
    p_values: list[float] = []
    for cause, effect in permutations(metrics, 2):
        x = window_data[cause].dropna().values
        y = window_data[effect].dropna().values
        min_len = min(len(x), len(y))
        if min_len < max_lag + 2:
            continue
        x = x[:min_len]
        y = y[:min_len]
        p, f = _granger_test_pair(x, y, max_lag)
        raw_edges.append(CausalEdge(cause=cause, effect=effect, p_value=p, f_statistic=f))
        p_values.append(p)

    if not raw_edges:
        return []

    reject, corrected_p, _, _ = multipletests(
        p_values, alpha=significance, method="fdr_bh"
    )

    significant_edges: list[CausalEdge] = []
    for edge, is_significant, p_corr in zip(raw_edges, reject, corrected_p):
        if is_significant:
            significant_edges.append(
                CausalEdge(
                    cause=edge.cause,
                    effect=edge.effect,
                    p_value=float(p_corr),
                    f_statistic=edge.f_statistic,
                )
            )
    return significant_edges
