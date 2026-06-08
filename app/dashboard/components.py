from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from streamlit_agraph import Node, Edge, Config

from app.analysis.contribution import Contribution
from app.analysis.root_cause import CausalGraph
from app.detection.base import AnomalyResult, AnomalyType

ANOMALY_FILL: dict[AnomalyType, str] = {
    AnomalyType.POINT: "rgba(255, 0, 0, 0.15)",
    AnomalyType.CONTEXTUAL: "rgba(255, 165, 0, 0.15)",
    AnomalyType.COLLECTIVE: "rgba(128, 0, 128, 0.15)",
}

ANOMALY_LINE: dict[AnomalyType, str] = {
    AnomalyType.POINT: "red",
    AnomalyType.CONTEXTUAL: "orange",
    AnomalyType.COLLECTIVE: "purple",
}


def _group_consecutive_timestamps(
    timestamps: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not timestamps:
        return []
    sorted_ts = sorted(timestamps)
    if len(sorted_ts) == 1:
        return [(sorted_ts[0], sorted_ts[0])]
    gaps = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    threshold = max(median_gap * 2, timedelta(seconds=1))
    regions: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = sorted_ts[0]
    end = sorted_ts[0]
    for i in range(1, len(sorted_ts)):
        if sorted_ts[i] - end <= threshold:
            end = sorted_ts[i]
        else:
            regions.append((start, end))
            start = sorted_ts[i]
            end = sorted_ts[i]
    regions.append((start, end))
    return regions


def render_time_series_chart(
    metrics_data: pd.DataFrame,
    anomalies: list[AnomalyResult],
    selected_metrics: list[str],
) -> go.Figure:
    fig = go.Figure()
    if metrics_data.empty or not selected_metrics:
        fig.update_layout(title="No data available")
        return fig
    for metric in selected_metrics:
        if metric not in metrics_data.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=metrics_data.index,
                y=metrics_data[metric],
                mode="lines",
                name=metric,
            )
        )
    anomaly_by_type: dict[AnomalyType, list[pd.Timestamp]] = {}
    for a in anomalies:
        if not a.is_anomaly:
            continue
        anomaly_by_type.setdefault(a.anomaly_type, []).append(a.timestamp)
    for atype, ts_list in anomaly_by_type.items():
        regions = _group_consecutive_timestamps(ts_list)
        for start_ts, end_ts in regions:
            pad = timedelta(seconds=0)
            fig.add_vrect(
                x0=start_ts - pad,
                x1=end_ts + pad,
                fillcolor=ANOMALY_FILL.get(atype, "rgba(255,0,0,0.15)"),
                line_width=0,
                layer="below",
                annotation_text=atype.value,
                annotation_position="top left",
                annotation_font_color=ANOMALY_LINE.get(atype, "red"),
                annotation_font_size=10,
            )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Metrics",
        height=500,
    )
    return fig


def render_anomaly_event_list(
    events: list[dict],
    on_confirm_tp: Callable[[int], None] | None = None,
    on_mark_fp: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=["ID", "Time", "Metric", "Severity", "Type", "Algorithm", "Confirmed"]
        )
    rows = []
    for e in events:
        severity = e.get("severity", 0.0)
        if severity >= 2.0:
            severity_label = "CRITICAL" if severity >= 3.0 else "HIGH"
        elif severity >= 1.0:
            severity_label = "MEDIUM"
        else:
            severity_label = "LOW"
        confirmed = ""
        if e.get("is_confirmed"):
            confirmed = e.get("confirmed_as", "TP")
        rows.append(
            {
                "ID": e.get("id", ""),
                "Time": e.get("start_time", ""),
                "Metric": e.get("metric_name", ""),
                "Severity": severity_label,
                "Type": e.get("anomaly_type", ""),
                "Algorithm": e.get("algorithm", ""),
                "Confirmed": confirmed,
            }
        )
    df = pd.DataFrame(rows)
    return df


def render_waterfall_chart(contributions: list[Contribution]) -> go.Figure:
    fig = go.Figure()
    if not contributions:
        fig.update_layout(title="No contribution data")
        return fig
    names = [c.sub_metric for c in contributions]
    values = [c.contribution_percentage for c in contributions]
    colors = [
        "#ef553b" if v > 30 else "#ffa15a" if v > 15 else "#636efa" for v in values
    ]
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Root Cause Contribution",
        xaxis_title="Contribution %",
        yaxis_title="Sub-metric",
        height=max(300, len(contributions) * 40),
    )
    return fig


def render_causal_graph(causal_graph: CausalGraph) -> dict:
    nodes = []
    for n in causal_graph.nodes:
        is_root = n == causal_graph.root_cause
        nodes.append(
            Node(
                id=n,
                label=n,
                size=30 if is_root else 20,
                color="#ef553b" if is_root else "#636efa",
            )
        )
    edges = []
    for source, target, data in causal_graph.edges:
        weight = data.get("weight", 1.0)
        edges.append(
            Edge(
                source=source,
                target=target,
                label=f"{weight:.2f}",
            )
        )
    config = Config(
        directed=True,
        physics=True,
        hierarchical=False,
    )
    return {"nodes": nodes, "edges": edges, "config": config}


def render_algorithm_table(performance_data: list[dict]) -> pd.DataFrame:
    if not performance_data:
        return pd.DataFrame(columns=["Algorithm", "TP", "FP", "Precision", "Recall"])
    rows = []
    for p in performance_data:
        rows.append(
            {
                "Algorithm": p.get("algorithm_name", ""),
                "TP": p.get("tp", 0),
                "FP": p.get("fp", 0),
                "Precision": round(p.get("precision", 0.0), 4),
                "Recall": round(p.get("recall", 0.0), 4),
            }
        )
    return pd.DataFrame(rows)
