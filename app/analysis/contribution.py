from dataclasses import dataclass
from itertools import permutations
from math import factorial

import numpy as np


@dataclass
class Contribution:
    sub_metric: str
    contribution_percentage: float


def _value_function(
    coalition: set[str],
    sub_metric_deltas: dict[str, float],
    aggregate_delta: float,
) -> float:
    if not coalition:
        return 0.0
    partial_sum = sum(sub_metric_deltas[m] for m in coalition)
    return abs(partial_sum - aggregate_delta)


def compute_shapley_contributions(
    sub_metric_deltas: dict[str, float],
    aggregate_delta: float,
) -> list[Contribution]:
    metrics = list(sub_metric_deltas.keys())
    n = len(metrics)
    if n == 0:
        return []
    shapley: dict[str, float] = {m: 0.0 for m in metrics}
    all_metrics = set(metrics)
    for m in metrics:
        others = all_metrics - {m}
        for perm in permutations(others):
            coalition_before: set[str] = set()
            for i, p in enumerate(perm):
                coalition_with = coalition_before | {m}
                v_with = _value_function(coalition_with, sub_metric_deltas, aggregate_delta)
                v_without = _value_function(coalition_before, sub_metric_deltas, aggregate_delta)
                shapley[m] += v_with - v_without
                coalition_before.add(p)
        shapley[m] /= factorial(n)
    total_shapley = sum(abs(v) for v in shapley.values())
    if total_shapley == 0:
        equal_pct = 100.0 / n
        return [Contribution(sub_metric=m, contribution_percentage=equal_pct) for m in metrics]
    contributions = [
        Contribution(
            sub_metric=m,
            contribution_percentage=round(abs(shapley[m]) / total_shapley * 100.0, 4),
        )
        for m in metrics
    ]
    contributions.sort(key=lambda c: c.contribution_percentage, reverse=True)
    return contributions


def compute_incremental_contributions(
    sub_metric_deltas: dict[str, float],
) -> list[Contribution]:
    metrics = list(sub_metric_deltas.keys())
    if not metrics:
        return []
    total_delta = sum(abs(d) for d in sub_metric_deltas.values())
    if total_delta == 0:
        equal_pct = 100.0 / len(metrics)
        return [Contribution(sub_metric=m, contribution_percentage=equal_pct) for m in metrics]
    contributions = [
        Contribution(
            sub_metric=m,
            contribution_percentage=round(abs(sub_metric_deltas[m]) / total_delta * 100.0, 4),
        )
        for m in metrics
    ]
    contributions.sort(key=lambda c: c.contribution_percentage, reverse=True)
    return contributions


def run_contribution_analysis(
    sub_metric_deltas: dict[str, float],
    aggregate_delta: float,
    method: str = "shapley",
) -> list[Contribution]:
    if method == "shapley":
        return compute_shapley_contributions(sub_metric_deltas, aggregate_delta)
    return compute_incremental_contributions(sub_metric_deltas)
