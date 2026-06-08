from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from app.detection.ensemble import EnsembleDetector
from app.detection.ml import IsolationForestDetector, LSTMEncoderDetector, ProphetDetector
from app.detection.statistical import IQRFecutor, STLDetector, ThreeSigmaDetector
from app.ingestion.parser import (
    parse_csv,
    parse_influxdb_lp,
    parse_prometheus_json,
    shard,
)
from app.storage.database import StorageManager

from .models import AnomalyItem, DataFormat, DetectionRequest, DetectionResult, TaskStatus

DETECTOR_MAP: dict[str, type] = {
    "three_sigma": ThreeSigmaDetector,
    "iqr": IQRFecutor,
    "stl": STLDetector,
    "isolation_forest": IsolationForestDetector,
    "lstm_autoencoder": LSTMEncoderDetector,
    "prophet": ProphetDetector,
}


@dataclass
class TaskState:
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[DetectionResult] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None


_tasks: dict[str, TaskState] = {}


def get_task(task_id: str) -> Optional[TaskState]:
    return _tasks.get(task_id)


def register_task(task_id: str) -> TaskState:
    state = TaskState()
    _tasks[task_id] = state
    return state


def _parse_data(request: DetectionRequest) -> tuple[pd.DataFrame, list[dict]]:
    if request.format == DataFormat.CSV:
        return parse_csv(request.data)
    elif request.format == DataFormat.PROMETHEUS:
        return parse_prometheus_json(request.data)
    elif request.format == DataFormat.INFLUXDB:
        return parse_influxdb_lp(request.data)
    else:
        raise ValueError(f"Unsupported format: {request.format}")


def _get_detectors(
    algorithms: list[str],
    metric_name: str,
    metric_configs: dict[str, dict],
) -> list:
    configs = metric_configs.get(metric_name, {})
    metric_algorithms = configs.get("algorithms", algorithms)
    if not metric_algorithms:
        metric_algorithms = list(DETECTOR_MAP.keys())
    detectors = []
    for algo_name in metric_algorithms:
        cls = DETECTOR_MAP.get(algo_name)
        if cls is not None:
            detectors.append(cls())
    return detectors


def _run_detection_for_metric(
    series: pd.Series,
    metric_name: str,
    request: DetectionRequest,
) -> list[AnomalyItem]:
    detectors = _get_detectors(request.algorithms, metric_name, request.metric_configs)
    if not detectors:
        return []

    algo_config = request.algorithm_configs.get(metric_name, {})
    metric_config = request.metric_configs.get(metric_name, {})
    merged_config = {**algo_config, **metric_config.get("params", {})}

    if len(detectors) > 1:
        ensemble = EnsembleDetector(detectors, weights=request.weights)
        results = ensemble.detect(series, {"ensemble_mode": request.ensemble_mode.value, **merged_config})
    else:
        results = detectors[0].detect(series, merged_config)

    items: list[AnomalyItem] = []
    for r in results:
        items.append(
            AnomalyItem(
                timestamp=str(r.timestamp),
                metric=metric_name,
                is_anomaly=r.is_anomaly,
                score=r.score,
                algorithm=r.algorithm_name,
                anomaly_type=r.anomaly_type.value,
            )
        )
    return items


async def run_detection_task(
    task_id: str,
    request: DetectionRequest,
    storage: StorageManager,
    ws_callback: Any = None,
) -> None:
    state = _tasks.get(task_id)
    if state is None:
        return

    state.status = TaskStatus.RUNNING

    try:
        df, metadata = await asyncio.to_thread(_parse_data, request)

        shards_list = await asyncio.to_thread(shard, df, metadata)

        all_anomalies: list[AnomalyItem] = []
        anomaly_metrics: list[str] = []

        for shard_df, shard_meta in shards_list:
            for meta in shard_meta:
                col_name = meta["name"]
                if col_name not in shard_df.columns:
                    continue
                series = shard_df[col_name].dropna()
                if series.empty:
                    continue
                items = await asyncio.to_thread(
                    _run_detection_for_metric, series, col_name, request
                )
                anomalous_items = [item for item in items if item.is_anomaly]
                all_anomalies.extend(anomalous_items)
                if anomalous_items:
                    anomaly_metrics.append(col_name)

                await storage.save_metric_data(col_name, shard_df[[col_name]])
                await storage.save_metric_metadata(meta)

        root_cause: Optional[dict] = None
        if len(anomaly_metrics) > 1:
            from app.analysis.root_cause import run_root_cause_analysis

            anomalous_df = df[anomaly_metrics]
            window = (0, len(anomalous_df) - 1)
            causal_graph = await asyncio.to_thread(
                run_root_cause_analysis,
                anomaly_metrics,
                anomalous_df,
                window,
            )
            root_cause = {
                "root_cause": causal_graph.root_cause,
                "nodes": causal_graph.nodes,
                "edges": [
                    {"cause": e[0], "effect": e[1], "data": e[2]}
                    for e in causal_graph.edges
                ],
            }

        alerts: list[dict] = []
        if all_anomalies:
            from app.alerts.aggregator import AlertAggregator, AggregatorConfig, AnomalyInput, SEVERITY_TO_FLOAT

            aggregator = AlertAggregator(AggregatorConfig())
            anomaly_inputs: list[AnomalyInput] = []
            for metric_name in set(item.metric for item in all_anomalies):
                metric_items = [item for item in all_anomalies if item.metric == metric_name]
                timestamps = [datetime.fromisoformat(item.timestamp) for item in metric_items]
                scores = [item.score for item in metric_items]
                anomaly_inputs.append(
                    AnomalyInput(
                        metric=metric_name,
                        timestamps=timestamps,
                        scores=scores,
                    )
                )
            alert_events = aggregator.process(anomaly_inputs)
            for event in alert_events:
                alert_dict = {
                    "id": event.id,
                    "root_metric": event.root_metric,
                    "related_metrics": event.related_metrics,
                    "start_time": event.start_time.isoformat(),
                    "end_time": event.end_time.isoformat(),
                    "severity": event.severity.value,
                    "suppressed": event.suppressed,
                    "suppression_reason": event.suppression_reason,
                    "channel": event.channel.value,
                    "details": event.details,
                }
                alerts.append(alert_dict)
                if not event.suppressed and ws_callback is not None:
                    try:
                        await ws_callback(alert_dict)
                    except Exception:
                        pass

                await storage.save_alert_event({
                    "root_metric": event.root_metric,
                    "related_metrics": str(event.related_metrics),
                    "start_time": event.start_time.isoformat(),
                    "severity": SEVERITY_TO_FLOAT.get(event.severity, 0.0),
                    "suppressed": event.suppressed,
                    "channel": event.channel.value,
                })

        for item in all_anomalies:
            event_id = await storage.save_anomaly_event({
                "metric_name": item.metric,
                "start_time": item.timestamp,
                "end_time": item.timestamp,
                "anomaly_type": item.anomaly_type,
                "severity": item.score,
                "algorithm": item.algorithm,
            })
            if ws_callback is not None and not any(a.get("suppressed", False) for a in alerts if a.get("root_metric") == item.metric):
                try:
                    await ws_callback({
                        "event_id": event_id,
                        "metric": item.metric,
                        "timestamp": item.timestamp,
                        "score": item.score,
                        "algorithm": item.algorithm,
                        "anomaly_type": item.anomaly_type,
                    })
                except Exception:
                    pass

        result = DetectionResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            anomalies=all_anomalies,
            root_cause=root_cause,
            alerts=alerts,
        )
        state.status = TaskStatus.COMPLETED
        state.result = result

    except Exception as e:
        state.status = TaskStatus.FAILED
        state.error = traceback.format_exc()
        state.result = DetectionResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
        )


async def run_batch_detection(
    task_ids: list[str],
    requests: list[DetectionRequest],
    storage: StorageManager,
    ws_callback: Any = None,
) -> None:
    tasks = [
        run_detection_task(tid, req, storage, ws_callback)
        for tid, req in zip(task_ids, requests)
    ]
    await asyncio.gather(*tasks)
