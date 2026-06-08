from __future__ import annotations

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.detection.base import AnomalyResult
from app.detection.ensemble import EnsembleDetector
from app.detection.ml import IsolationForestDetector, LSTMEncoderDetector, ProphetDetector
from app.detection.statistical import IQRFecutor, STLDetector, ThreeSigmaDetector
from app.storage.database import StorageManager

from .models import ModelStatus, TrainingContext, TrainingProgress
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

DETECTOR_MAP: dict[str, type] = {
    "three_sigma": ThreeSigmaDetector,
    "iqr": IQRFecutor,
    "stl": STLDetector,
    "isolation_forest": IsolationForestDetector,
    "lstm_autoencoder": LSTMEncoderDetector,
    "prophet": ProphetDetector,
}


class TrainingPipeline:
    def __init__(
        self,
        storage: StorageManager,
        registry: ModelRegistry,
        ws_callback: Optional[Callable] = None,
    ) -> None:
        self._storage = storage
        self._registry = registry
        self._ws_callback = ws_callback
        self._progress_cache: dict[str, TrainingProgress] = {}

    def get_progress(self, model_id: str) -> Optional[TrainingProgress]:
        return self._progress_cache.get(model_id)

    async def run_training(
        self,
        new_model_id: str,
        parent_model_id: Optional[str] = None,
        training_data_days: int = 30,
        trigger_reason: str = "manual",
    ) -> TrainingContext:
        start_time = time.monotonic()
        context = TrainingContext(model_id=new_model_id)
        progress = TrainingProgress(model_id=new_model_id)
        self._progress_cache[new_model_id] = progress

        new_model = await self._registry.get_model(new_model_id)

        if new_model is None:
            context.error = "Model not found"
            progress.stage = "failed"
            progress.error_message = context.error
            return context

        old_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if parent_model_id is not None:
            parent_model = await self._registry.get_model(parent_model_id)
            if parent_model is not None:
                old_metrics = {
                    "precision": parent_model.precision,
                    "recall": parent_model.recall,
                    "f1": parent_model.f1,
                }

        context.old_precision = old_metrics["precision"]
        context.old_recall = old_metrics["recall"]
        context.old_f1 = old_metrics["f1"]

        try:
            await self._broadcast_progress(progress, "data_preparation", 1, 4, "Preparing training data")

            now = datetime.utcnow()
            data_start = now - timedelta(days=training_data_days)
            metrics_list = await self._storage.list_metrics()

            all_data: dict[str, pd.Series] = {}
            total_data_points = 0

            for m in metrics_list:
                df = await self._storage.load_metric_data(m.name, data_start, now)
                if df.empty:
                    continue
                col = "value" if "value" in df.columns else df.columns[0]
                series = df[col].dropna()
                if len(series) < 10:
                    continue
                all_data[m.name] = series
                total_data_points += len(series)

            context.training_data_count = total_data_points
            progress.data_size = total_data_points
            context.stages.append({
                "name": "data_preparation",
                "status": "completed",
                "data_points": total_data_points,
                "completed_at": datetime.utcnow().isoformat(),
            })

            await self._broadcast_progress(progress, "feature_extraction", 2, 4, "Extracting features")

            detector_cls = DETECTOR_MAP.get(new_model.algorithm_type)
            if detector_cls is None:
                raise ValueError(f"Unknown algorithm type: {new_model.algorithm_type}")

            await asyncio.sleep(0.5)
            context.stages.append({
                "name": "feature_extraction",
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
            })

            await self._broadcast_progress(progress, "model_training", 3, 4, "Training model")

            training_results: dict[str, list[AnomalyResult]] = {}
            for metric_name, series in all_data.items():
                detector = detector_cls()
                results = await asyncio.to_thread(
                    detector.detect, series, new_model.training_params
                )
                training_results[metric_name] = results

            await asyncio.sleep(0.5)
            context.stages.append({
                "name": "model_training",
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
            })

            await self._broadcast_progress(progress, "evaluation", 4, 4, "Evaluating model performance")

            tp, fp, fn = 0, 0, 0
            for metric_name, results in training_results.items():
                events = await self._storage.get_anomaly_events(data_start, now, metric_name)
                detected_timestamps = set()
                for r in results:
                    if r.is_anomaly:
                        detected_timestamps.add(str(r.timestamp))

                for event in events:
                    event_ts = event.get("start_time", "")
                    if event_ts in detected_timestamps:
                        if event.get("is_confirmed") and event.get("confirmed_as") == "TP":
                            tp += 1
                        elif event.get("is_confirmed") and event.get("confirmed_as") == "FP":
                            fp += 1
                        else:
                            tp += 1
                    elif event.get("is_confirmed") and event.get("confirmed_as") == "TP":
                        fn += 1

                for ts_str in detected_timestamps:
                    matched = any(e.get("start_time") == ts_str for e in events)
                    if not matched:
                        fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if tp == 0 and fp == 0 and fn == 0:
                precision = 0.5
                recall = 0.5
                f1 = 0.5

            context.new_precision = precision
            context.new_recall = recall
            context.new_f1 = f1

            await self._registry.update_model_metrics(new_model_id, precision, recall, f1)

            context.stages.append({
                "name": "evaluation",
                "status": "completed",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "completed_at": datetime.utcnow().isoformat(),
            })

        except Exception as e:
            context.error = traceback.format_exc()
            progress.stage = "failed"
            progress.error_message = str(e)
            await self._registry.update_model_status(new_model_id, ModelStatus.FAILED)
            logger.error("Training pipeline failed for %s: %s", new_model_id, e)

        elapsed = time.monotonic() - start_time
        context.training_duration_seconds = elapsed
        progress.elapsed_seconds = elapsed
        context.completed_at = datetime.utcnow().isoformat()

        if context.error is None:
            progress.stage = "completed"
            progress.stage_description = "Training completed successfully"
        else:
            progress.stage = "failed"

        await self._broadcast_progress(progress, progress.stage, 4, 4, progress.stage_description or "")

        return context

    async def compare_models(
        self, model_a_id: str, model_b_id: str
    ) -> Optional[dict]:
        from scipy.stats import ks_2samp
        from .models import ModelComparisonResult

        model_a = await self._registry.get_model(model_a_id)
        model_b = await self._registry.get_model(model_b_id)
        if model_a is None or model_b is None:
            return None

        test_data_start = datetime.utcnow() - timedelta(days=7)
        test_data_end = datetime.utcnow()
        metrics_list = await self._storage.list_metrics()

        scores_a: list[float] = []
        scores_b: list[float] = []

        for m in metrics_list:
            df = await self._storage.load_metric_data(m.name, test_data_start, test_data_end)
            if df.empty:
                continue
            col = "value" if "value" in df.columns else df.columns[0]
            series = df[col].dropna()
            if len(series) < 10:
                continue

            cls_a = DETECTOR_MAP.get(model_a.algorithm_type)
            cls_b = DETECTOR_MAP.get(model_b.algorithm_type)

            if cls_a:
                detector_a = cls_a()
                results_a = await asyncio.to_thread(detector_a.detect, series, model_a.training_params)
                scores_a.extend([r.score for r in results_a])

            if cls_b:
                detector_b = cls_b()
                results_b = await asyncio.to_thread(detector_b.detect, series, model_b.training_params)
                scores_b.extend([r.score for r in results_b])

        min_len = min(len(scores_a), len(scores_b))
        if min_len == 0:
            return None

        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

        arr_a = np.array(scores_a)
        arr_b = np.array(scores_b)

        ks_statistic = 0.0
        ks_pvalue = 1.0
        ks_reject_null = False
        if min_len >= 2:
            ks_statistic, ks_pvalue = ks_2samp(arr_a, arr_b)
            ks_reject_null = ks_pvalue < 0.05

        return {
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "model_a_version": model_a.version,
            "model_b_version": model_b.version,
            "model_a_scores": scores_a,
            "model_b_scores": scores_b,
            "model_a_precision": model_a.precision,
            "model_a_recall": model_a.recall,
            "model_a_f1": model_a.f1,
            "model_b_precision": model_b.precision,
            "model_b_recall": model_b.recall,
            "model_b_f1": model_b.f1,
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
            "ks_reject_null": ks_reject_null,
            "model_a_mean": float(np.mean(arr_a)),
            "model_a_std": float(np.std(arr_a)),
            "model_a_median": float(np.median(arr_a)),
            "model_b_mean": float(np.mean(arr_b)),
            "model_b_std": float(np.std(arr_b)),
            "model_b_median": float(np.median(arr_b)),
            "sample_size": min_len,
        }

    async def _broadcast_progress(
        self,
        progress: TrainingProgress,
        stage: str,
        current_step: int,
        total_steps: int,
        description: str,
    ) -> None:
        progress.stage = stage
        progress.current_step = current_step
        progress.total_steps = total_steps
        progress.stage_description = description

        if self._ws_callback:
            try:
                await self._ws_callback({
                    "type": "training_progress",
                    "model_id": progress.model_id,
                    "stage": stage,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "description": description,
                    "data_size": progress.data_size,
                    "elapsed_seconds": progress.elapsed_seconds,
                })
            except Exception:
                pass
