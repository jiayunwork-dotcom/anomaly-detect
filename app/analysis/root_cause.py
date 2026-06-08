from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from app.analysis.contribution import Contribution, run_contribution_analysis
from app.analysis.correlation import run_correlation_analysis, CorrelationResult
from app.analysis.granger import run_granger_causality, CausalEdge


@dataclass
class CausalGraph:
    nodes: list[str]
    edges: list[tuple[str, str, dict]]
    root_cause: str | None
    contributions: list[Contribution] = field(default_factory=list)


def _build_causal_graph(
    metrics: list[str],
    causal_edges: list[CausalEdge],
) -> nx.DiGraph:
    g = nx.DiGraph()
    for m in metrics:
        g.add_node(m)
    for edge in causal_edges:
        weight = edge.f_statistic / (edge.p_value + 1e-12)
        g.add_edge(edge.cause, edge.effect, weight=weight, p_value=edge.p_value, f_statistic=edge.f_statistic)
    return g


def _identify_root_cause(g: nx.DiGraph) -> str | None:
    if g.number_of_nodes() == 0:
        return None
    if g.number_of_edges() == 0:
        nodes = list(g.nodes())
        return nodes[0]
    candidates: dict[str, float] = {}
    for node in g.nodes():
        out_degree = g.out_degree(node)
        out_weight = sum(g[node][succ].get("weight", 0.0) for succ in g.successors(node))
        candidates[node] = out_degree + out_weight
    return max(candidates, key=candidates.get)


def run_root_cause_analysis(
    anomaly_metrics: list[str],
    time_series_data: pd.DataFrame,
    anomaly_window: tuple[int, int],
    aggregate_metrics: dict[str, dict[str, float]] | None = None,
    pearson_threshold: float = 0.7,
    dtw_threshold: float = 1.0,
    max_lag_window: int = 60,
    granger_max_lag: int = 5,
    granger_significance: float = 0.05,
    contribution_method: str = "shapley",
) -> CausalGraph:
    metric_data = time_series_data[anomaly_metrics]

    correlation_result: CorrelationResult = run_correlation_analysis(
        data=metric_data,
        anomaly_window=anomaly_window,
        pearson_threshold=pearson_threshold,
        dtw_threshold=dtw_threshold,
        max_lag_window=max_lag_window,
    )

    causal_edges: list[CausalEdge] = run_granger_causality(
        data=metric_data,
        anomaly_window=anomaly_window,
        max_lag=granger_max_lag,
        significance=granger_significance,
    )

    contributions: list[Contribution] = []
    if aggregate_metrics:
        start, end = anomaly_window
        window_data = metric_data.iloc[start:end]
        for agg_name, sub_deltas in aggregate_metrics.items():
            agg_series = window_data.get(agg_name)
            if agg_series is not None:
                baseline_val = agg_series.iloc[0]
                anomalous_val = agg_series.iloc[-1]
                agg_delta = anomalous_val - baseline_val
            else:
                agg_delta = sum(sub_deltas.values())
            contribs = run_contribution_analysis(
                sub_metric_deltas=sub_deltas,
                aggregate_delta=agg_delta,
                method=contribution_method,
            )
            contributions.extend(contribs)

    g = _build_causal_graph(anomaly_metrics, causal_edges)
    root_cause = _identify_root_cause(g)

    edges_list: list[tuple[str, str, dict]] = []
    for u, v, data in g.edges(data=True):
        edges_list.append((u, v, dict(data)))

    return CausalGraph(
        nodes=list(g.nodes()),
        edges=edges_list,
        root_cause=root_cause,
        contributions=contributions,
    )
