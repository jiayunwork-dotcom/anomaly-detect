from __future__ import annotations

import asyncio
import json
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
from app.scheduler.models import ScheduleCreateRequest, ScheduleInterval
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

        st.header("Schedule")
        sched_name = st.text_input("Task Name", value="Untitled Schedule", key="sched_name")
        interval_options = {
            "1 minute": ScheduleInterval.MIN_1,
            "5 minutes": ScheduleInterval.MIN_5,
            "15 minutes": ScheduleInterval.MIN_15,
            "30 minutes": ScheduleInterval.MIN_30,
            "1 hour": ScheduleInterval.HOUR_1,
            "Custom Cron": ScheduleInterval.CUSTOM,
        }
        sched_interval_label = st.selectbox(
            "Interval", list(interval_options.keys()), key="sched_interval"
        )
        sched_interval = interval_options[sched_interval_label]
        cron_expr = None
        if sched_interval == ScheduleInterval.CUSTOM:
            cron_expr = st.text_input(
                "Cron Expression (min hour day month day_of_week)",
                value="*/5 * * * *",
                key="sched_cron",
            )

        available_metrics = filtered if filtered else metric_names
        sched_metrics = st.multiselect(
            "Metrics to Monitor", available_metrics, key="sched_metrics"
        )
        sched_algorithms = st.multiselect(
            "Algorithms",
            list(ALGORITHM_REGISTRY.keys()),
            default=["three_sigma", "iqr"],
            key="sched_algorithms",
        )
        sched_ensemble = st.radio(
            "Voting Mode", ["majority", "weighted"],
            horizontal=True, key="sched_ensemble"
        )
        sched_weights: dict[str, float] = {}
        if sched_ensemble == "weighted":
            for algo_name in sched_algorithms:
                w = st.slider(
                    algo_name, min_value=0.0, max_value=2.0,
                    value=1.0, step=0.1, key=f"sched_weight_{algo_name}",
                )
                sched_weights[algo_name] = w

        if st.button("Save Schedule", key="save_schedule_btn"):
            if not sched_metrics:
                st.warning("Select at least one metric to monitor.")
            elif not sched_algorithms:
                st.warning("Select at least one algorithm.")
            else:
                import httpx
                try:
                    resp = httpx.post(
                        "http://localhost:8000/api/schedules",
                        json={
                            "name": sched_name,
                            "interval": sched_interval.value,
                            "cron_expression": cron_expr,
                            "metrics": sched_metrics,
                            "algorithms": sched_algorithms,
                            "ensemble_mode": sched_ensemble,
                            "weights": sched_weights,
                        },
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        st.success(f"Schedule created: {resp.json().get('id', '')}")
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"API error: {e}")

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


def render_tab_scheduled_tasks() -> None:
    import httpx

    st.header("Scheduled Tasks")
    schedules: list[dict] = []
    try:
        resp = httpx.get("http://localhost:8000/api/schedules", timeout=10.0)
        if resp.status_code == 200:
            schedules = resp.json()
    except Exception as e:
        st.error(f"Failed to load schedules: {e}")
        return

    if not schedules:
        st.info("No scheduled tasks found. Create one from the sidebar Schedule section.")
        return

    status_icons = {
        "running": "🟢 Running",
        "paused": "🟡 Paused",
        "failed": "🔴 Failed",
    }
    interval_labels = {
        "1m": "1 min", "5m": "5 min", "15m": "15 min",
        "30m": "30 min", "1h": "1 hour", "custom": "Custom",
    }

    for sched in schedules:
        sched_id = sched.get("id", "")
        status = sched.get("status", "running")
        status_label = status_icons.get(status, status)
        interval = sched.get("interval", "")
        interval_label = interval_labels.get(interval, interval)

        with st.expander(
            f"{sched.get('name', 'Untitled')} — {status_label} — {interval_label}",
            expanded=False,
        ):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            col1.metric("Status", status_label)
            col2.metric("Next Run", sched.get("next_run_time", "N/A"))
            last_result = f"{sched.get('last_anomaly_count', 0)} anomalies"
            if sched.get("last_alert_triggered"):
                last_result += " (alert fired)"
            col3.metric("Last Result", last_result)
            col4.metric("Last Run", sched.get("last_run_time", "Never"))

            st.caption(
                f"Metrics: {', '.join(sched.get('metrics', []))} | "
                f"Algorithms: {', '.join(sched.get('algorithms', []))} | "
                f"Voting: {sched.get('ensemble_mode', 'majority')}"
            )
            if sched.get("cron_expression"):
                st.caption(f"Cron: {sched['cron_expression']}")

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            if status == "running":
                if btn_col1.button("Pause", key=f"pause_{sched_id}"):
                    try:
                        r = httpx.put(
                            f"http://localhost:8000/api/schedules/{sched_id}/pause",
                            timeout=10.0,
                        )
                        if r.status_code == 200:
                            st.success("Paused")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(f"Error: {e}")
            elif status == "paused":
                if btn_col1.button("Resume", key=f"resume_{sched_id}"):
                    try:
                        r = httpx.put(
                            f"http://localhost:8000/api/schedules/{sched_id}/resume",
                            timeout=10.0,
                        )
                        if r.status_code == 200:
                            st.success("Resumed")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(f"Error: {e}")

            if btn_col3.button("Delete", key=f"delete_{sched_id}"):
                try:
                    r = httpx.delete(
                        f"http://localhost:8000/api/schedules/{sched_id}",
                        timeout=10.0,
                    )
                    if r.status_code == 200:
                        st.success("Deleted")
                        st.rerun()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(f"Error: {e}")

            st.subheader("Execution History (Last 20)")
            try:
                hist_resp = httpx.get(
                    f"http://localhost:8000/api/schedules/{sched_id}/history",
                    timeout=10.0,
                )
                if hist_resp.status_code == 200:
                    history = hist_resp.json()
                    if history:
                        hist_rows = []
                        for h in history:
                            hist_status = h.get("status", "")
                            if hist_status == "completed":
                                status_icon = "✅"
                            elif hist_status == "failed":
                                status_icon = "❌"
                            else:
                                status_icon = "⏳"
                            hist_rows.append({
                                "Start": h.get("start_time", ""),
                                "End": h.get("end_time", "—"),
                                "Status": f"{status_icon} {hist_status}",
                                "Anomalies": h.get("anomaly_count", 0),
                                "Alert": "Yes" if h.get("alert_triggered") else "No",
                                "Error": (h.get("error", "") or "")[:80],
                            })
                        st.dataframe(
                            pd.DataFrame(hist_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No execution history yet.")
                else:
                    st.warning("Failed to load history.")
            except Exception as e:
                st.warning(f"History load error: {e}")


def render_tab_model_registry() -> None:
    import httpx
    from scipy.stats import gaussian_kde

    API_BASE = "http://localhost:8000"

    st.header("Model Registry")

    if "pending_dismiss_alerts" not in st.session_state:
        st.session_state.pending_dismiss_alerts = []

    try:
        alerts_resp = httpx.get(f"{API_BASE}/api/models/alerts", params={"dismissed": "false"}, timeout=10.0)
        if alerts_resp.status_code == 200:
            active_alerts = alerts_resp.json()
        else:
            active_alerts = []
    except Exception:
        active_alerts = []

    if st.session_state.pending_dismiss_alerts:
        for dismiss_id in st.session_state.pending_dismiss_alerts:
            try:
                httpx.put(f"{API_BASE}/api/models/alerts/{dismiss_id}/dismiss", timeout=10.0)
            except Exception:
                pass
        st.session_state.pending_dismiss_alerts = []
        try:
            alerts_resp = httpx.get(f"{API_BASE}/api/models/alerts", params={"dismissed": "false"}, timeout=10.0)
            if alerts_resp.status_code == 200:
                active_alerts = alerts_resp.json()
            else:
                active_alerts = []
        except Exception:
            active_alerts = []

    if active_alerts:
        st.markdown("#### ⚠️ Model Performance Degradation Alerts")
        for alert in active_alerts:
            alert_id = alert.get("id", "")
            alert_cols = st.columns([4, 1])
            suggestion_label = "🔄 Trigger Retrain" if alert.get("suggestion") == "trigger_retrain" else "🔍 Check Data Quality"
            with alert_cols[0]:
                st.warning(
                    f"**{alert.get('model_name', '')}** — F1: {alert.get('current_f1', 0):.3f} "
                    f"(threshold: {alert.get('f1_threshold', 0):.3f}) — "
                    f"Consecutive low windows: {alert.get('consecutive_low_windows', 0)} — "
                    f"Suggestion: {suggestion_label}"
                )
            with alert_cols[1]:
                if st.button("Dismiss", key=f"dismiss_alert_{alert_id}"):
                    st.session_state.pending_dismiss_alerts.append(alert_id)
                    st.rerun()

    col_reg1, col_reg2 = st.columns([3, 1])
    with col_reg2:
        with st.expander("Register New Model", expanded=False):
            new_name = st.text_input("Model Name", value="", key="mr_new_name")
            new_algo = st.selectbox(
                "Algorithm Type",
                list(ALGORITHM_REGISTRY.keys()),
                key="mr_new_algo",
            )
            new_params_str = st.text_area(
                "Training Params (JSON)",
                value="{}",
                key="mr_new_params",
                height=80,
            )
            if st.button("Register & Train", key="mr_register_btn"):
                if not new_name.strip():
                    st.warning("Model name is required.")
                else:
                    try:
                        params = json.loads(new_params_str) if new_params_str.strip() else {}
                    except Exception:
                        st.error("Invalid JSON for training params.")
                        params = {}
                    try:
                        now = datetime.utcnow()
                        resp = httpx.post(
                            f"{API_BASE}/api/models",
                            json={
                                "name": new_name.strip(),
                                "algorithm_type": new_algo,
                                "training_params": params,
                                "training_data_start": (now - timedelta(days=30)).isoformat(),
                                "training_data_end": now.isoformat(),
                            },
                            timeout=15.0,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"Model registered: {data.get('id', '')} (v{data.get('version', '')})")
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"API error: {e}")

    models: list[dict] = []
    try:
        resp = httpx.get(f"{API_BASE}/api/models", timeout=10.0)
        if resp.status_code == 200:
            models = resp.json()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    if not models:
        st.info("No models registered yet. Use the form above to register a new model.")
        return

    status_labels = {
        "training": "● Training",
        "active": "✓ Active",
        "retired": "○ Retired",
        "failed": "✗ Failed",
    }

    for model_group in models:
        name = model_group.get("name", "")
        algo = model_group.get("algorithm_type", "")
        version_count = model_group.get("version_count", 0)
        active_f1 = model_group.get("active_f1", 0.0)
        active_version = model_group.get("active_version", "")
        active_id = model_group.get("active_model_id")
        active_status = "✓ Active" if active_id else "○ Retired"

        with st.expander(
            f"**{name}** — {algo} — v{active_version} — {active_status} — F1: {active_f1:.3f} — {version_count} version(s)",
            expanded=False,
        ):
            versions: list[dict] = []
            try:
                vresp = httpx.get(f"{API_BASE}/api/models/{name}/versions", timeout=10.0)
                if vresp.status_code == 200:
                    versions = vresp.json()
            except Exception:
                st.warning("Failed to load versions.")

            if versions:
                st.subheader("Version Timeline")
                timeline_data = []
                for v in versions:
                    v_status = v.get("status", "training")
                    status_display = status_labels.get(v_status, v_status)
                    timeline_data.append({
                        "Version": v.get("version", ""),
                        "Status": status_display,
                        "Precision": f"{v.get('precision', 0.0):.4f}",
                        "Recall": f"{v.get('recall', 0.0):.4f}",
                        "F1": f"{v.get('f1', 0.0):.4f}",
                        "Created": v.get("created_at", "")[:19],
                        "ID": v.get("id", ""),
                    })
                st.dataframe(
                    pd.DataFrame(timeline_data),
                    use_container_width=True,
                    hide_index=True,
                )

            if active_id:
                st.subheader("Retraining Configuration")
                retrain_config: Optional[dict] = None
                try:
                    rcresp = httpx.get(
                        f"{API_BASE}/api/models/{name}/retrain-config",
                        timeout=10.0,
                    )
                    if rcresp.status_code == 200:
                        retrain_config = rcresp.json()
                except Exception:
                    pass

                rc_trigger = st.selectbox(
                    "Trigger Type",
                    ["scheduled", "performance", "data_drift"],
                    index=["scheduled", "performance", "data_drift"].index(
                        retrain_config.get("trigger_type", "scheduled") if retrain_config else "scheduled"
                    ),
                    key=f"mr_trigger_{name}",
                )
                rc_col1, rc_col2 = st.columns(2)
                with rc_col1:
                    if rc_trigger == "scheduled":
                        rc_interval = st.number_input(
                            "Interval (hours)",
                            min_value=1,
                            max_value=720,
                            value=retrain_config.get("scheduled_interval_hours", 24) if retrain_config else 24,
                            key=f"mr_interval_{name}",
                        )
                    else:
                        rc_interval = 24

                    if rc_trigger == "performance":
                        rc_window = st.number_input(
                            "Window Size (detection windows)",
                            min_value=1,
                            max_value=100,
                            value=retrain_config.get("performance_window_size", 10) if retrain_config else 10,
                            key=f"mr_perf_window_{name}",
                        )
                        rc_f1_thresh = st.number_input(
                            "F1 Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=retrain_config.get("performance_f1_threshold", 0.7) if retrain_config else 0.7,
                            step=0.05,
                            key=f"mr_f1_thresh_{name}",
                        )
                    else:
                        rc_window = 10
                        rc_f1_thresh = 0.7

                with rc_col2:
                    if rc_trigger == "data_drift":
                        rc_kl_thresh = st.number_input(
                            "KL Divergence Threshold",
                            min_value=0.01,
                            max_value=10.0,
                            value=retrain_config.get("drift_kl_threshold", 0.5) if retrain_config else 0.5,
                            step=0.05,
                            key=f"mr_kl_thresh_{name}",
                        )
                    else:
                        rc_kl_thresh = 0.5

                    rc_data_days = st.number_input(
                        "Training Data (days)",
                        min_value=1,
                        max_value=365,
                        value=retrain_config.get("training_data_days", 30) if retrain_config else 30,
                        key=f"mr_data_days_{name}",
                    )

                rc_enabled = st.checkbox(
                    "Enabled",
                    value=retrain_config.get("enabled", True) if retrain_config else True,
                    key=f"mr_enabled_{name}",
                )

                if st.button("Save Retrain Config", key=f"mr_save_config_{name}"):
                    try:
                        save_resp = httpx.post(
                            f"{API_BASE}/api/models/retrain-config",
                            json={
                                "model_name": name,
                                "trigger_type": rc_trigger,
                                "scheduled_interval_hours": rc_interval,
                                "performance_window_size": rc_window,
                                "performance_f1_threshold": rc_f1_thresh,
                                "drift_kl_threshold": rc_kl_thresh,
                                "training_data_days": rc_data_days,
                                "enabled": rc_enabled,
                            },
                            timeout=10.0,
                        )
                        if save_resp.status_code == 200:
                            st.success("Retrain config saved.")
                        else:
                            st.error(f"Failed: {save_resp.text}")
                    except Exception as e:
                        st.error(f"API error: {e}")

                st.subheader("Actions")
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("Trigger Retrain", key=f"mr_retrain_{active_id}"):
                        try:
                            tr_resp = httpx.post(
                                f"{API_BASE}/api/models/{active_id}/retrain",
                                timeout=15.0,
                            )
                            if tr_resp.status_code == 200:
                                st.success(f"Retraining triggered: {tr_resp.json().get('new_model_id', '')}")
                                st.rerun()
                            else:
                                st.error(f"Failed: {tr_resp.text}")
                        except Exception as e:
                            st.error(f"API error: {e}")

                with action_col2:
                    if st.button("Retire", key=f"mr_retire_{active_id}"):
                        try:
                            ret_resp = httpx.put(
                                f"{API_BASE}/api/models/{active_id}/retire",
                                timeout=10.0,
                            )
                            if ret_resp.status_code == 200:
                                st.success("Model retired.")
                                st.rerun()
                            else:
                                st.error(f"Failed: {ret_resp.text}")
                        except Exception as e:
                            st.error(f"API error: {e}")

                with action_col3:
                    pass

            for v in versions:
                v_id = v.get("id", "")
                v_status = v.get("status", "")
                if v_status == "retired":
                    if st.button(f"Delete v{v.get('version', '')}", key=f"mr_del_{v_id}"):
                        try:
                            del_resp = httpx.delete(
                                f"{API_BASE}/api/models/{v_id}",
                                timeout=10.0,
                            )
                            if del_resp.status_code == 200:
                                st.success("Deleted.")
                                st.rerun()
                            else:
                                st.error(f"Failed: {del_resp.text}")
                        except Exception as e:
                            st.error(f"API error: {e}")
                    break

            st.subheader("Training Progress")
            for v in versions:
                v_id = v.get("id", "")
                v_status = v.get("status", "")
                try:
                    prog_resp = httpx.get(
                        f"{API_BASE}/api/models/{v_id}/progress",
                        timeout=10.0,
                    )
                    if prog_resp.status_code == 200:
                        prog = prog_resp.json()
                        stage = prog.get("stage", "idle")
                        if stage not in ("idle", "completed", "failed"):
                            current = prog.get("current_step", 0)
                            total = prog.get("total_steps", 4)
                            desc = prog.get("stage_description", stage)
                            st.progress(current / total if total > 0 else 0)
                            st.caption(f"v{v.get('version', '')} — {desc} ({current}/{total})")
                        elif stage == "failed":
                            st.error(f"v{v.get('version', '')} — Training failed: {prog.get('error_message', 'Unknown')}")
                except Exception:
                    pass

            st.subheader("Training History")
            try:
                for v in versions:
                    v_id = v.get("id", "")
                    hist_resp = httpx.get(
                        f"{API_BASE}/api/models/{v_id}/training-history",
                        timeout=10.0,
                    )
                    if hist_resp.status_code == 200:
                        histories = hist_resp.json()
                        if histories:
                            for h in histories:
                                h_col1, h_col2 = st.columns([3, 2])
                                with h_col1:
                                    st.markdown(
                                        f"**v{v.get('version', '')}** — "
                                        f"Data: {h.get('training_data_count', 0)} pts — "
                                        f"Duration: {h.get('training_duration_seconds', 0):.1f}s"
                                    )
                                    if h.get("error"):
                                        st.error(f"Error: {h['error'][:200]}")
                                with h_col2:
                                    if h.get("new_f1", 0) > 0:
                                        st.markdown(
                                            f"Old P/R/F1: {h.get('old_precision', 0):.3f} / "
                                            f"{h.get('old_recall', 0):.3f} / {h.get('old_f1', 0):.3f}"
                                        )
                                        st.markdown(
                                            f"New P/R/F1: {h.get('new_precision', 0):.3f} / "
                                            f"{h.get('new_recall', 0):.3f} / {h.get('new_f1', 0):.3f}"
                                        )
                                        if h.get("auto_activated"):
                                            st.success("Auto-activated ✓")
                                        else:
                                            st.warning("Not activated")
            except Exception as e:
                st.warning(f"History load error: {e}")

    st.header("A/B Testing")
    ab_tests: list[dict] = []
    try:
        ab_resp = httpx.get(f"{API_BASE}/api/models/ab-tests", timeout=10.0)
        if ab_resp.status_code == 200:
            ab_tests = ab_resp.json()
    except Exception:
        pass

    running_tests = [t for t in ab_tests if t.get("status") == "running"]
    completed_tests = [t for t in ab_tests if t.get("status") != "running"]

    with st.expander("Start New A/B Test", expanded=False):
        ab_model_names = [m.get("name", "") for m in models]
        if len(ab_model_names) < 1:
            st.info("No models available for A/B testing.")
        else:
            ab_model_name = st.selectbox("Model Name", ab_model_names, key="ab_model_name")
            ab_versions: list[dict] = []
            try:
                ab_vresp = httpx.get(f"{API_BASE}/api/models/{ab_model_name}/versions", timeout=10.0)
                if ab_vresp.status_code == 200:
                    ab_versions = ab_vresp.json()
            except Exception:
                pass

            if len(ab_versions) < 2:
                st.info("Need at least 2 versions to start an A/B test.")
            else:
                ab_ver_labels = [
                    f"v{v.get('version', '')} ({v.get('status', '')}) F1={v.get('f1', 0):.3f} — {v.get('id', '')}"
                    for v in ab_versions
                ]
                ab_ver_ids = [v.get("id", "") for v in ab_versions]

                ab_col1, ab_col2 = st.columns(2)
                with ab_col1:
                    ab_sel_primary = st.selectbox("Primary Model", range(len(ab_ver_labels)), format_func=lambda i: ab_ver_labels[i], key="ab_primary")
                with ab_col2:
                    ab_sel_challenger = st.selectbox("Challenger Model", range(len(ab_ver_labels)), format_func=lambda i: ab_ver_labels[i], key="ab_challenger")

                ab_traffic = st.slider("Primary Traffic %", min_value=10, max_value=90, value=80, step=5, key="ab_traffic")
                ab_min_windows = st.number_input("Min Detection Windows", min_value=1, max_value=100, value=5, key="ab_min_windows")
                ab_f1_threshold = st.number_input("F1 Improvement Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01, key="ab_f1_thresh")

                if st.button("Start A/B Test", key="ab_start_btn"):
                    if ab_sel_primary == ab_sel_challenger:
                        st.warning("Select different models for primary and challenger.")
                    else:
                        try:
                            ab_start_resp = httpx.post(
                                f"{API_BASE}/api/models/ab-test",
                                json={
                                    "model_name": ab_model_name,
                                    "primary_model_id": ab_ver_ids[ab_sel_primary],
                                    "challenger_model_id": ab_ver_ids[ab_sel_challenger],
                                    "primary_traffic_pct": float(ab_traffic),
                                    "min_windows": ab_min_windows,
                                    "f1_improvement_threshold": ab_f1_threshold,
                                },
                                timeout=15.0,
                            )
                            if ab_start_resp.status_code == 200:
                                st.success("A/B test started!")
                                st.rerun()
                            else:
                                st.error(f"Failed: {ab_start_resp.text}")
                        except Exception as e:
                            st.error(f"API error: {e}")

    if running_tests:
        st.subheader("Running A/B Tests")
        for t in running_tests:
            t_name = t.get("model_name", "")
            t_status = t.get("status", "")
            t_windows = t.get("windows_completed", 0)
            t_min = t.get("min_windows", 5)
            t_primary_f1 = t.get("primary_f1", 0.0)
            t_challenger_f1 = t.get("challenger_f1", 0.0)
            t_traffic = t.get("primary_traffic_pct", 80.0)

            with st.container():
                st.markdown(
                    f"**{t_name}** — {t_status.upper()} — "
                    f"Windows: {t_windows}/{t_min} — "
                    f"Traffic: Primary {t_traffic}% / Challenger {100 - t_traffic}% — "
                    f"Primary F1: {t_primary_f1:.3f} / Challenger F1: {t_challenger_f1:.3f}"
                )
                progress_pct = min(t_windows / t_min, 1.0) if t_min > 0 else 0
                st.progress(progress_pct)
                f1_diff = t_challenger_f1 - t_primary_f1
                if f1_diff > 0:
                    st.info(f"Challenger is leading by +{f1_diff:.4f} F1")
                else:
                    st.info(f"Primary is leading by +{-f1_diff:.4f} F1")

                if st.button("Stop A/B Test", key=f"ab_stop_{t_name}"):
                    try:
                        stop_resp = httpx.delete(f"{API_BASE}/api/models/{t_name}/ab-test", timeout=10.0)
                        if stop_resp.status_code == 200:
                            st.success("A/B test stopped")
                            st.rerun()
                        else:
                            st.error(f"Failed: {stop_resp.text}")
                    except Exception as e:
                        st.error(f"API error: {e}")

    if completed_tests:
        st.subheader("Completed A/B Tests")
        for t in completed_tests:
            t_name = t.get("model_name", "")
            t_status = t.get("status", "")
            t_primary_f1 = t.get("primary_f1", 0.0)
            t_challenger_f1 = t.get("challenger_f1", 0.0)
            result_icon = "✅ Challenger Promoted" if t_status == "completed_promoted" else "❌ Challenger Retired"
            st.markdown(
                f"**{t_name}** — {result_icon} — "
                f"Primary F1: {t_primary_f1:.3f} / Challenger F1: {t_challenger_f1:.3f} — "
                f"Ended: {t.get('ended_at', '')[:19]}"
            )

    st.header("Model Version Comparison")
    all_versions: list[dict] = []
    try:
        for mg in models:
            vresp = httpx.get(f"{API_BASE}/api/models/{mg['name']}/versions", timeout=10.0)
            if vresp.status_code == 200:
                all_versions.extend(vresp.json())
    except Exception:
        pass

    if len(all_versions) < 2:
        st.info("Need at least 2 model versions to compare.")
    else:
        version_labels = [
            f"{v.get('name', '')} v{v.get('version', '')} ({v.get('status', '')}) — {v.get('id', '')}"
            for v in all_versions
        ]
        version_ids = [v.get("id", "") for v in all_versions]

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            sel_a = st.selectbox("Model A", range(len(version_labels)), format_func=lambda i: version_labels[i], key="mr_cmp_a")
        with cmp_col2:
            sel_b = st.selectbox("Model B", range(len(version_labels)), format_func=lambda i: version_labels[i], key="mr_cmp_b")

        if st.button("Compare", key="mr_compare_btn"):
            if sel_a == sel_b:
                st.warning("Please select two different models.")
            else:
                model_a_id = version_ids[sel_a]
                model_b_id = version_ids[sel_b]
                try:
                    cmp_resp = httpx.post(
                        f"{API_BASE}/api/models/compare",
                        json={"model_a_id": model_a_id, "model_b_id": model_b_id},
                        timeout=30.0,
                    )
                    if cmp_resp.status_code == 200:
                        cmp_data = cmp_resp.json()
                        scores_a = cmp_data.get("model_a_scores", [])
                        scores_b = cmp_data.get("model_b_scores", [])
                        sample_size = cmp_data.get("sample_size", 0)
                        ver_a = f"v{cmp_data.get('model_a_version', 'A')}"
                        ver_b = f"v{cmp_data.get('model_b_version', 'B')}"

                        if not scores_a or not scores_b:
                            st.info("No score data available for comparison.")
                        else:
                            sample_size_a = cmp_data.get("sample_size_a", len(scores_a))
                            sample_size_b = cmp_data.get("sample_size_b", len(scores_b))
                            min_sample = min(sample_size_a, sample_size_b)
                            if min_sample < 50:
                                st.warning("⚠️ Sample size insufficient (<50), statistical test results may be unreliable.")

                            st.subheader("Kolmogorov-Smirnov Test Report")
                            ks_stat = cmp_data.get("ks_statistic", 0.0)
                            ks_pval = cmp_data.get("ks_pvalue", 1.0)
                            ks_reject = cmp_data.get("ks_reject_null", False)

                            ks_col1, ks_col2, ks_col3 = st.columns(3)
                            ks_col1.metric("KS Statistic", f"{ks_stat:.4f}")
                            ks_col2.metric("p-value", f"{ks_pval:.6f}")
                            if ks_reject:
                                ks_col3.metric("H₀: Same Distribution", "Rejected ✗", delta="p < 0.05")
                            else:
                                ks_col3.metric("H₀: Same Distribution", "Not Rejected ✓", delta="p ≥ 0.05")

                            if ks_reject:
                                st.info("The two score distributions are significantly different at the 0.05 level.")
                            else:
                                st.info("Cannot reject the null hypothesis that the two distributions are the same at the 0.05 level.")

                            st.subheader("Distribution Statistics Comparison")
                            stats_data = {
                                "Statistic": ["Mean", "Std Dev", "Median", "Sample Size"],
                                ver_a: [
                                    f"{cmp_data.get('model_a_mean', 0):.4f}",
                                    f"{cmp_data.get('model_a_std', 0):.4f}",
                                    f"{cmp_data.get('model_a_median', 0):.4f}",
                                    str(sample_size_a),
                                ],
                                ver_b: [
                                    f"{cmp_data.get('model_b_mean', 0):.4f}",
                                    f"{cmp_data.get('model_b_std', 0):.4f}",
                                    f"{cmp_data.get('model_b_median', 0):.4f}",
                                    str(sample_size_b),
                                ],
                            }
                            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

                            st.subheader("Kernel Density Estimation (Overlay)")
                            arr_a = np.array(scores_a)
                            arr_b = np.array(scores_b)
                            x_min = min(arr_a.min(), arr_b.min())
                            x_max = max(arr_a.max(), arr_b.max())
                            x_range = x_max - x_min
                            if x_range > 0:
                                x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 300)
                                kde_fig = go.Figure()
                                try:
                                    kde_a = gaussian_kde(arr_a)
                                    kde_b = gaussian_kde(arr_b)
                                    kde_fig.add_trace(
                                        go.Scatter(
                                            x=x_grid,
                                            y=kde_a(x_grid),
                                            mode="lines",
                                            name=f"{ver_a} (n={sample_size_a})",
                                            line=dict(color="#636efa", width=2),
                                            fill="tozeroy",
                                            opacity=0.3,
                                        )
                                    )
                                    kde_fig.add_trace(
                                        go.Scatter(
                                            x=x_grid,
                                            y=kde_b(x_grid),
                                            mode="lines",
                                            name=f"{ver_b} (n={sample_size_b})",
                                            line=dict(color="#ef553b", width=2),
                                            fill="tozeroy",
                                            opacity=0.3,
                                        )
                                    )
                                except Exception:
                                    kde_fig.add_trace(
                                        go.Histogram(x=arr_a, name=ver_a, opacity=0.5, histnorm="probability density")
                                    )
                                    kde_fig.add_trace(
                                        go.Histogram(x=arr_b, name=ver_b, opacity=0.5, histnorm="probability density")
                                    )
                                kde_fig.update_layout(
                                    title="Anomaly Score Distribution (KDE)",
                                    xaxis_title="Score",
                                    yaxis_title="Density",
                                    height=400,
                                )
                                st.plotly_chart(kde_fig, use_container_width=True)
                            else:
                                st.info("Score range is zero, cannot compute KDE.")

                            st.subheader("Anomaly Score Time Series")
                            score_fig = go.Figure()
                            step_a = max(1, len(scores_a) // 500)
                            step_b = max(1, len(scores_b) // 500)
                            idx_a = list(range(0, len(scores_a), step_a))
                            idx_b = list(range(0, len(scores_b), step_b))
                            sampled_a = [scores_a[i] for i in idx_a]
                            sampled_b = [scores_b[i] for i in idx_b]

                            score_fig.add_trace(
                                go.Scatter(
                                    x=idx_a,
                                    y=sampled_a,
                                    mode="lines",
                                    name=ver_a,
                                    line=dict(color="#636efa"),
                                )
                            )
                            score_fig.add_trace(
                                go.Scatter(
                                    x=idx_b,
                                    y=sampled_b,
                                    mode="lines",
                                    name=ver_b,
                                    line=dict(color="#ef553b"),
                                )
                            )
                            score_fig.update_layout(
                                title="Anomaly Score Comparison",
                                xaxis_title="Sample Index",
                                yaxis_title="Score",
                                height=400,
                            )
                            st.plotly_chart(score_fig, use_container_width=True)

                            st.subheader("Metrics Comparison")
                            metrics_fig = go.Figure()
                            categories = ["Precision", "Recall", "F1"]
                            vals_a = [
                                cmp_data.get("model_a_precision", 0),
                                cmp_data.get("model_a_recall", 0),
                                cmp_data.get("model_a_f1", 0),
                            ]
                            vals_b = [
                                cmp_data.get("model_b_precision", 0),
                                cmp_data.get("model_b_recall", 0),
                                cmp_data.get("model_b_f1", 0),
                            ]
                            metrics_fig.add_trace(
                                go.Bar(
                                    name=ver_a,
                                    x=categories,
                                    y=vals_a,
                                    marker_color="#636efa",
                                )
                            )
                            metrics_fig.add_trace(
                                go.Bar(
                                    name=ver_b,
                                    x=categories,
                                    y=vals_b,
                                    marker_color="#ef553b",
                                )
                            )
                            metrics_fig.update_layout(
                                title="Precision / Recall / F1 Comparison",
                                yaxis_title="Score",
                                barmode="group",
                                height=400,
                            )
                            st.plotly_chart(metrics_fig, use_container_width=True)
                    else:
                        st.error(f"Comparison failed: {cmp_resp.text}")
                except Exception as e:
                    st.error(f"API error: {e}")


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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Time Series", "Anomaly Events", "Root Cause Analysis",
         "Algorithm Performance", "Data Import", "Scheduled Tasks", "Model Registry"]
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

    with tab6:
        render_tab_scheduled_tasks()

    with tab7:
        render_tab_model_registry()


if __name__ == "__main__":
    main()
