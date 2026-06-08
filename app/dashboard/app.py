from __future__ import annotations

import asyncio
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_agraph import agraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.analysis.root_cause import CausalGraph, run_root_cause_analysis
from app.detection.base import AnomalyResult, AnomalyType, BaseDetector
from app.detection.ensemble import EnsembleDetector
from app.detection.granularity import (
    CollectiveAnomalyDetector,
    ContextualAnomalyDetector,
    PointAnomalyDetector,
)
from app.detection.ml import IsolationForestDetector, LSTMEncoderDetector, ProphetDetector
from app.detection.statistical import IQRFecutor, STLDetector, ThreeSigmaDetector
from app.ingestion.parser import parse_csv, parse_influxdb_lp, parse_prometheus_json
from app.labeling.loop import LabelingLoop
from app.storage.database import StorageManager

from app.dashboard.components import (
    render_algorithm_table,
    render_anomaly_event_list,
    render_causal_graph,
    render_time_series_chart,
    render_waterfall_chart,
)

_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True)
        t.start()
    return _loop


def run_async(coro: Any) -> Any:
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def init_storage() -> StorageManager:
    if "storage" not in st.session_state:
        storage = StorageManager()
        run_async(storage.init())
        st.session_state.storage = storage
    return st.session_state.storage


def get_time_range(
    preset: str,
    custom_start: datetime | None,
    custom_end: datetime | None,
) -> tuple[datetime, datetime]:
    now = datetime.utcnow()
    ranges: dict[str, tuple[datetime, datetime]] = {
        "Last 1h": (now - timedelta(hours=1), now),
        "Last 6h": (now - timedelta(hours=6), now),
        "Last 24h": (now - timedelta(hours=24), now),
        "Last 7d": (now - timedelta(days=7), now),
    }
    if preset == "Custom":
        if custom_start and custom_end:
            return custom_start, custom_end
        return now - timedelta(hours=1), now
    return ranges.get(preset, ranges["Last 1h"])


ALGORITHM_REGISTRY: dict[str, type[BaseDetector]] = {
    "three_sigma": ThreeSigmaDetector,
    "iqr": IQRFecutor,
    "stl": STLDetector,
    "isolation_forest": IsolationForestDetector,
    "lstm_autoencoder": LSTMEncoderDetector,
    "prophet": ProphetDetector,
}

GRANULARITY_REGISTRY: dict[str, str] = {
    "point": "Point Anomaly",
    "contextual": "Contextual Anomaly",
    "collective": "Collective Anomaly",
}


def run_detection(
    metrics_data: pd.DataFrame,
    selected_metrics: list[str],
    selected_algorithms: list[str],
    selected_granularities: list[str],
    ensemble_mode: str,
    weights: dict[str, float],
) -> list[AnomalyResult]:
    base_detectors: list[BaseDetector] = []
    for algo_name in selected_algorithms:
        cls = ALGORITHM_REGISTRY.get(algo_name)
        if cls is not None:
            base_detectors.append(cls())

    detectors: list[BaseDetector] = []
    for g in selected_granularities:
        if g == "point":
            for d in base_detectors:
                detectors.append(PointAnomalyDetector(d))
        elif g == "contextual":
            detectors.append(ContextualAnomalyDetector())
        elif g == "collective":
            detectors.append(CollectiveAnomalyDetector())

    if not detectors:
        detectors = [PointAnomalyDetector(ThreeSigmaDetector())]

    ensemble = EnsembleDetector(detectors, weights=weights if ensemble_mode == "weighted" else None)
    all_results: list[AnomalyResult] = []
    for metric in selected_metrics:
        if metric not in metrics_data.columns:
            continue
        series = metrics_data[metric].dropna()
        if len(series) < 3:
            continue
        results = ensemble.detect(series, {"ensemble_mode": ensemble_mode})
        all_results.extend(results)
    return all_results


def load_metrics_data(
    storage: StorageManager,
    metric_names: list[str],
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    combined: pd.DataFrame = pd.DataFrame()
    for metric in metric_names:
        df = run_async(storage.load_metric_data(metric, start_time, end_time))
        if df.empty:
            continue
        if "value" in df.columns:
            df = df.rename(columns={"value": metric})
        elif len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: metric})
        if metric in df.columns:
            if combined.empty:
                combined = df[[metric]]
            else:
                combined = combined.join(df[[metric]], how="outer")
    if not combined.empty:
        combined = combined.sort_index()
    return combined


def render_sidebar() -> tuple[list[str], datetime, datetime, list[str], list[str], str, dict[str, float]]:
    with st.sidebar:
        st.header("Filters")
        time_preset = st.selectbox(
            "Time Range",
            ["Last 1h", "Last 6h", "Last 24h", "Last 7d", "Custom"],
        )
        custom_start: datetime | None = None
        custom_end: datetime | None = None
        if time_preset == "Custom":
            cs = st.date_input("Start Date", value=datetime.utcnow().date() - timedelta(days=1))
            ce = st.date_input("End Date", value=datetime.utcnow().date())
            if cs:
                custom_start = datetime.combine(cs, datetime.min.time())
            if ce:
                custom_end = datetime.combine(ce, datetime.max.time())
        start_time, end_time = get_time_range(time_preset, custom_start, custom_end)

        storage = init_storage()
        metrics_list = run_async(storage.list_metrics())
        metric_names = [m.name for m in metrics_list]

        search_query = st.text_input("Search Metrics")
        filtered = (
            [n for n in metric_names if search_query.lower() in n.lower()]
            if search_query
            else metric_names
        )
        label_filter = st.text_input("Filter by Label (key=value)")
        if label_filter:
            filtered = [n for n in filtered if label_filter.lower() in n.lower()]

        selected_metrics = st.multiselect("Select Metrics", filtered)

        st.header("Detection Config")
        selected_algorithms = st.multiselect(
            "Algorithms",
            list(ALGORITHM_REGISTRY.keys()),
            default=["three_sigma", "iqr"],
        )
        selected_granularities = st.multiselect(
            "Granularity",
            list(GRANULARITY_REGISTRY.keys()),
            default=["point", "contextual", "collective"],
            format_func=lambda x: GRANULARITY_REGISTRY[x],
        )
        ensemble_mode = st.radio("Voting Mode", ["majority", "weighted"], horizontal=True)

        weights: dict[str, float] = {}
        if ensemble_mode == "weighted":
            st.subheader("Algorithm Weights")
            for algo_name in selected_algorithms:
                w = st.slider(
                    algo_name,
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"weight_{algo_name}",
                )
                weights[algo_name] = w

        st.header("Data Import")
        st.file_uploader("Upload File", type=["csv", "json", "txt"], key="sidebar_upload")
        st.selectbox("Format", ["CSV", "Prometheus JSON", "InfluxDB LP"], key="sidebar_format")

    return selected_metrics, start_time, end_time, selected_algorithms, selected_granularities, ensemble_mode, weights


def render_tab_timeseries(
    metrics_data: pd.DataFrame,
    anomalies: list[AnomalyResult],
    selected_metrics: list[str],
) -> None:
    if not selected_metrics:
        st.info("Select metrics from the sidebar to view time series data.")
        return
    if metrics_data.empty:
        st.info("No data found for the selected metrics and time range.")
        return
    metric_select = st.multiselect(
        "Display metrics",
        selected_metrics,
        default=selected_metrics,
        key="ts_metric_select",
    )
    if not metric_select:
        st.info("Select at least one metric to display.")
        return
    filtered_anomalies = [
        a for a in anomalies if any(
            getattr(a, "_metric_name", None) == m for m in metric_select
        )
    ] if any(hasattr(a, "_metric_name") for a in anomalies) else anomalies
    fig = render_time_series_chart(metrics_data, filtered_anomalies, metric_select)
    st.plotly_chart(fig, use_container_width=True)
    anomaly_count = sum(1 for a in anomalies if a.is_anomaly)
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalies Detected", anomaly_count)
    point_count = sum(1 for a in anomalies if a.is_anomaly and a.anomaly_type == AnomalyType.POINT)
    ctx_count = sum(1 for a in anomalies if a.is_anomaly and a.anomaly_type == AnomalyType.CONTEXTUAL)
    col_count = sum(1 for a in anomalies if a.is_anomaly and a.anomaly_type == AnomalyType.COLLECTIVE)
    col2.metric("Point", point_count)
    col3.metric("Contextual / Collective", ctx_count + col_count)


def render_tab_events(
    storage: StorageManager,
    labeling_loop: LabelingLoop,
    start_time: datetime,
    end_time: datetime,
) -> None:
    events = run_async(storage.get_anomaly_events(start_time, end_time))
    events_sorted = sorted(events, key=lambda e: e.get("start_time", ""), reverse=True)
    if not events_sorted:
        st.info("No anomaly events found for the selected time range.")
        return
    df = render_anomaly_event_list(events_sorted)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Confirm / Mark Events")
    for event in events_sorted[:50]:
        event_id = event.get("id")
        if not event_id:
            continue
        severity = event.get("severity", 0.0)
        if severity >= 3.0:
            color = "🔴"
        elif severity >= 2.0:
            color = "🟠"
        elif severity >= 1.0:
            color = "🟡"
        else:
            color = "🟢"
        col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
        col1.text(str(event.get("start_time", "")))
        col2.text(str(event.get("metric_name", "")))
        col3.markdown(f"{color} {severity:.1f}")
        if event.get("is_confirmed"):
            confirmed_as = event.get("confirmed_as", "")
            label = "✅ TP" if confirmed_as == "TP" else "❌ FP"
            col4.markdown(label)
            col5.write("")
        else:
            if col4.button("Confirm TP", key=f"tp_{event_id}"):
                run_async(labeling_loop.confirm_event(event_id))
                st.rerun()
            if col5.button("Mark FP", key=f"fp_{event_id}"):
                run_async(labeling_loop.mark_false_alarm(event_id))
                st.rerun()


def render_tab_rootcause(
    metrics_data: pd.DataFrame,
    anomalies: list[AnomalyResult],
    selected_metrics: list[str],
) -> None:
    if not selected_metrics or metrics_data.empty:
        st.info("Select metrics and ensure data is loaded to run root cause analysis.")
        return
    anomaly_indices = [i for i, a in enumerate(anomalies) if a.is_anomaly]
    if not anomaly_indices:
        st.info("No anomalies detected to analyze.")
        return
    anomaly_window = (min(anomaly_indices), max(anomaly_indices) + 1)
    with st.spinner("Running root cause analysis..."):
        try:
            causal_graph: CausalGraph = run_root_cause_analysis(
                anomaly_metrics=selected_metrics,
                time_series_data=metrics_data[selected_metrics].dropna(),
                anomaly_window=anomaly_window,
            )
        except Exception as e:
            st.error(f"Root cause analysis failed: {e}")
            return
    if causal_graph.contributions:
        wf_fig = render_waterfall_chart(causal_graph.contributions)
        st.plotly_chart(wf_fig, use_container_width=True)
    else:
        st.info("No contribution data available.")
    if causal_graph.nodes:
        st.subheader("Causal Graph")
        graph_data = render_causal_graph(causal_graph)
        agraph(**graph_data)
        if causal_graph.root_cause:
            st.metric("Identified Root Cause", causal_graph.root_cause)
    else:
        st.info("No causal graph data available.")


def render_tab_performance(labeling_loop: LabelingLoop) -> None:
    perf_data = run_async(labeling_loop.compute_algorithm_performance())
    perf_list = [
        {"algorithm_name": algo, **stats} for algo, stats in perf_data.items()
    ]
    if not perf_list:
        st.info(
            "No algorithm performance data available. "
            "Confirm or mark anomaly events to populate."
        )
        return
    df = render_algorithm_table(perf_list)
    st.dataframe(df, use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[p["algorithm_name"] for p in perf_list],
            y=[p["precision"] for p in perf_list],
            name="Precision",
            marker_color="#636efa",
        )
    )
    fig.update_layout(
        title="Algorithm Precision Comparison",
        xaxis_title="Algorithm",
        yaxis_title="Precision",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tab_import(storage: StorageManager) -> None:
    st.header("Data Import")
    uploaded = st.file_uploader(
        "Upload CSV / Prometheus JSON / InfluxDB LP",
        type=["csv", "json", "txt"],
        key="import_tab",
    )
    fmt = st.selectbox(
        "Format",
        ["CSV", "Prometheus JSON", "InfluxDB LP"],
        key="import_format_tab",
    )
    if uploaded is None:
        return
    content = uploaded.getvalue().decode("utf-8")
    try:
        if fmt == "CSV":
            df, metadata = parse_csv(content)
        elif fmt == "Prometheus JSON":
            df, metadata = parse_prometheus_json(content)
        else:
            df, metadata = parse_influxdb_lp(content)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    if st.button("Import to Storage", key="import_btn"):
        progress = st.progress(0)
        total = len(df.columns)
        for idx, col in enumerate(df.columns):
            col_df = df[[col]].copy()
            col_df.columns = ["value"]
            run_async(storage.save_metric_data(col, col_df))
            for meta in metadata:
                if meta["name"] != col:
                    continue
                valid_range = meta.get("valid_range", (None, None))
                v_min = valid_range[0]
                v_max = valid_range[1]
                if v_min is not None and isinstance(v_min, float) and np.isnan(v_min):
                    v_min = None
                if v_max is not None and isinstance(v_max, float) and np.isnan(v_max):
                    v_max = None
                meta_dict = {
                    "name": meta["name"],
                    "unit": meta.get("unit", ""),
                    "source": meta.get("source", ""),
                    "frequency": 0.0,
                    "valid_min": v_min,
                    "valid_max": v_max,
                }
                run_async(storage.save_metric_metadata(meta_dict))
            progress.progress((idx + 1) / total)
        st.success(f"Imported {total} metrics successfully!")
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Anomaly Detection Dashboard",
        layout="wide",
    )
    st.title("Time Series Anomaly Detection")

    storage = init_storage()
    labeling_loop = LabelingLoop(storage)

    selected_metrics, start_time, end_time, selected_algorithms, selected_granularities, ensemble_mode, weights = render_sidebar()

    metrics_data: pd.DataFrame = pd.DataFrame()
    anomalies: list[AnomalyResult] = []

    if selected_metrics:
        metrics_data = load_metrics_data(storage, selected_metrics, start_time, end_time)
        if not metrics_data.empty:
            anomalies = run_detection(
                metrics_data, selected_metrics,
                selected_algorithms, selected_granularities,
                ensemble_mode, weights,
            )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Time Series", "Anomaly Events", "Root Cause Analysis",
         "Algorithm Performance", "Data Import"]
    )

    with tab1:
        render_tab_timeseries(metrics_data, anomalies, selected_metrics)

    with tab2:
        render_tab_events(storage, labeling_loop, start_time, end_time)

    with tab3:
        render_tab_rootcause(metrics_data, anomalies, selected_metrics)

    with tab4:
        render_tab_performance(labeling_loop)

    with tab5:
        render_tab_import(storage)


if __name__ == "__main__":
    main()
