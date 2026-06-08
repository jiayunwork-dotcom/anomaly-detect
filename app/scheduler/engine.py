from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.detection.ensemble import EnsembleDetector
from app.detection.ml import IsolationForestDetector, LSTMEncoderDetector, ProphetDetector
from app.detection.statistical import IQRFecutor, STLDetector, ThreeSigmaDetector
from app.storage.database import StorageManager

from .models import ScheduleCreateRequest, ScheduleInterval, ScheduleStatus

logger = logging.getLogger(__name__)

DETECTOR_MAP: dict[str, type] = {
    "three_sigma": ThreeSigmaDetector,
    "iqr": IQRFecutor,
    "stl": STLDetector,
    "isolation_forest": IsolationForestDetector,
    "lstm_autoencoder": LSTMEncoderDetector,
    "prophet": ProphetDetector,
}

INTERVAL_MAP: dict[ScheduleInterval, dict] = {
    ScheduleInterval.MIN_1: {"minutes": 1},
    ScheduleInterval.MIN_5: {"minutes": 5},
    ScheduleInterval.MIN_15: {"minutes": 15},
    ScheduleInterval.MIN_30: {"minutes": 30},
    ScheduleInterval.HOUR_1: {"hours": 1},
}


class ScheduleEntry:
    def __init__(
        self,
        schedule_id: str,
        config: ScheduleCreateRequest,
        status: ScheduleStatus = ScheduleStatus.RUNNING,
        last_run_time: Optional[str] = None,
        last_anomaly_count: int = 0,
        last_alert_triggered: bool = False,
        created_at: Optional[str] = None,
    ):
        self.schedule_id = schedule_id
        self.config = config
        self.status = status
        self.last_run_time = last_run_time
        self.last_anomaly_count = last_anomaly_count
        self.last_alert_triggered = last_alert_triggered
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.last_execution_start: Optional[str] = None


class SchedulerEngine:
    def __init__(
        self,
        storage: StorageManager,
        ws_callback: Optional[Callable] = None,
    ):
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._storage = storage
        self._ws_callback = ws_callback
        self._schedules: dict[str, ScheduleEntry] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._load_schedules_from_db()
        self._scheduler.start()
        logger.info("SchedulerEngine started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("SchedulerEngine stopped")

    async def _load_schedules_from_db(self) -> None:
        schedules = await self._storage.get_all_schedules()
        for s in schedules:
            schedule_id = s["id"]
            config = ScheduleCreateRequest(
                name=s.get("name", "Untitled"),
                interval=ScheduleInterval(s["interval"]),
                cron_expression=s.get("cron_expression"),
                metrics=json.loads(s.get("metrics", "[]")),
                algorithms=json.loads(s.get("algorithms", '["three_sigma","iqr"]')),
                ensemble_mode=s.get("ensemble_mode", "majority"),
                weights=json.loads(s.get("weights", "{}")),
            )
            status = ScheduleStatus(s.get("status", "running"))
            entry = ScheduleEntry(
                schedule_id=schedule_id,
                config=config,
                status=status,
                last_run_time=s.get("last_run_time"),
                last_anomaly_count=s.get("last_anomaly_count", 0),
                last_alert_triggered=s.get("last_alert_triggered", False),
                created_at=s.get("created_at"),
            )
            self._schedules[schedule_id] = entry
            if status == ScheduleStatus.RUNNING:
                self._add_job(schedule_id, config)

    def _add_job(self, schedule_id: str, config: ScheduleCreateRequest) -> None:
        job_id = f"schedule_{schedule_id}"
        if config.interval == ScheduleInterval.CUSTOM and config.cron_expression:
            parts = config.cron_expression.strip().split()
            if len(parts) == 5:
                trigger = CronTrigger(
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                    timezone="UTC",
                )
            else:
                trigger = IntervalTrigger(minutes=5)
        else:
            interval_kwargs = INTERVAL_MAP.get(config.interval, {"minutes": 5})
            trigger = IntervalTrigger(**interval_kwargs)

        self._scheduler.add_job(
            self._run_scheduled_detection,
            trigger=trigger,
            id=job_id,
            args=[schedule_id],
            replace_existing=True,
        )

    def _remove_job(self, schedule_id: str) -> None:
        job_id = f"schedule_{schedule_id}"
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass

    async def create_schedule(self, config: ScheduleCreateRequest) -> str:
        schedule_id = uuid4().hex[:12]
        entry = ScheduleEntry(schedule_id=schedule_id, config=config)
        self._schedules[schedule_id] = entry

        await self._storage.save_schedule({
            "id": schedule_id,
            "name": config.name,
            "interval": config.interval.value,
            "cron_expression": config.cron_expression,
            "metrics": json.dumps(config.metrics),
            "algorithms": json.dumps(config.algorithms),
            "ensemble_mode": config.ensemble_mode,
            "weights": json.dumps(config.weights),
            "status": ScheduleStatus.RUNNING.value,
            "created_at": entry.created_at,
        })

        self._add_job(schedule_id, config)
        logger.info("Created schedule %s", schedule_id)
        return schedule_id

    async def delete_schedule(self, schedule_id: str) -> bool:
        if schedule_id not in self._schedules:
            return False
        self._remove_job(schedule_id)
        del self._schedules[schedule_id]
        await self._storage.delete_schedule(schedule_id)
        logger.info("Deleted schedule %s", schedule_id)
        return True

    async def pause_schedule(self, schedule_id: str) -> bool:
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return False
        if entry.status != ScheduleStatus.RUNNING:
            return False
        entry.status = ScheduleStatus.PAUSED
        self._remove_job(schedule_id)
        await self._storage.update_schedule_status(schedule_id, ScheduleStatus.PAUSED.value)
        logger.info("Paused schedule %s", schedule_id)
        return True

    async def resume_schedule(self, schedule_id: str) -> bool:
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return False
        if entry.status != ScheduleStatus.PAUSED:
            return False
        entry.status = ScheduleStatus.RUNNING
        self._add_job(schedule_id, entry.config)
        await self._storage.update_schedule_status(schedule_id, ScheduleStatus.RUNNING.value)
        logger.info("Resumed schedule %s", schedule_id)
        return True

    def list_schedules(self) -> list[dict]:
        result = []
        for schedule_id, entry in self._schedules.items():
            next_run = None
            job_id = f"schedule_{schedule_id}"
            try:
                job = self._scheduler.get_job(job_id)
                if job and job.next_run_time:
                    next_run = job.next_run_time.isoformat()
            except Exception:
                pass
            result.append({
                "id": schedule_id,
                "name": entry.config.name,
                "interval": entry.config.interval.value,
                "cron_expression": entry.config.cron_expression,
                "metrics": entry.config.metrics,
                "algorithms": entry.config.algorithms,
                "ensemble_mode": entry.config.ensemble_mode,
                "weights": entry.config.weights,
                "status": entry.status.value,
                "next_run_time": next_run,
                "last_run_time": entry.last_run_time,
                "last_anomaly_count": entry.last_anomaly_count,
                "last_alert_triggered": entry.last_alert_triggered,
                "created_at": entry.created_at,
            })
        return result

    def get_schedule(self, schedule_id: str) -> Optional[dict]:
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return None
        schedules = self.list_schedules()
        for s in schedules:
            if s["id"] == schedule_id:
                return s
        return None

    async def _run_scheduled_detection(self, schedule_id: str) -> None:
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return

        start_time = datetime.utcnow()
        entry.last_execution_start = start_time.isoformat()

        history_id = await self._storage.save_execution_history({
            "schedule_id": schedule_id,
            "start_time": start_time.isoformat(),
            "status": "running",
        })

        try:
            end_time = datetime.utcnow()
            if entry.last_run_time:
                window_start = datetime.fromisoformat(entry.last_run_time)
            else:
                window_start = end_time - timedelta(hours=1)

            all_anomaly_count = 0
            alert_triggered = False

            for metric_name in entry.config.metrics:
                df = await self._storage.load_metric_data(metric_name, window_start, end_time)
                if df.empty or metric_name not in df.columns:
                    continue
                series = df[metric_name].dropna()
                if len(series) < 3:
                    continue

                detectors = []
                for algo_name in entry.config.algorithms:
                    cls = DETECTOR_MAP.get(algo_name)
                    if cls is not None:
                        detectors.append(cls())

                if not detectors:
                    continue

                if len(detectors) > 1:
                    weights = entry.config.weights if entry.config.ensemble_mode == "weighted" else None
                    ensemble = EnsembleDetector(detectors, weights=weights)
                    results = await asyncio.to_thread(
                        ensemble.detect, series,
                        {"ensemble_mode": entry.config.ensemble_mode},
                    )
                else:
                    results = await asyncio.to_thread(detectors[0].detect, series, {})

                anomaly_items = [r for r in results if r.is_anomaly]
                all_anomaly_count += len(anomaly_items)

                for item in anomaly_items:
                    await self._storage.save_anomaly_event({
                        "metric_name": metric_name,
                        "start_time": str(item.timestamp),
                        "end_time": str(item.timestamp),
                        "anomaly_type": item.anomaly_type.value,
                        "severity": item.score,
                        "algorithm": item.algorithm_name,
                    })

                if anomaly_items:
                    alert_triggered = True
                    from app.alerts.aggregator import AlertAggregator, AggregatorConfig, AnomalyInput
                    aggregator = AlertAggregator(AggregatorConfig())
                    anomaly_inputs: list[AnomalyInput] = []
                    timestamps = [item.timestamp for item in anomaly_items]
                    scores = [item.score for item in anomaly_items]
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
                        await self._storage.save_alert_event({
                            "root_metric": event.root_metric,
                            "related_metrics": str(event.related_metrics),
                            "start_time": event.start_time.isoformat(),
                            "severity": float(event.severity.value == "critical"),
                            "suppressed": event.suppressed,
                            "channel": event.channel.value,
                        })
                        if not event.suppressed and self._ws_callback is not None:
                            try:
                                await self._ws_callback(alert_dict)
                            except Exception:
                                pass

                    if self._ws_callback is not None:
                        try:
                            await self._ws_callback({
                                "type": "schedule_detection",
                                "schedule_id": schedule_id,
                                "metric": metric_name,
                                "anomaly_count": len(anomaly_items),
                            })
                        except Exception:
                            pass

            entry.last_run_time = end_time.isoformat()
            entry.last_anomaly_count = all_anomaly_count
            entry.last_alert_triggered = alert_triggered
            entry.status = ScheduleStatus.RUNNING

            await self._storage.update_execution_history(history_id, {
                "end_time": end_time.isoformat(),
                "anomaly_count": all_anomaly_count,
                "alert_triggered": alert_triggered,
                "status": "completed",
            })
            await self._storage.update_schedule_last_run(
                schedule_id,
                end_time.isoformat(),
                all_anomaly_count,
                alert_triggered,
            )

        except Exception as e:
            entry.status = ScheduleStatus.FAILED
            error_msg = traceback.format_exc()
            logger.error("Schedule %s execution failed: %s", schedule_id, e)

            await self._storage.update_execution_history(history_id, {
                "end_time": datetime.utcnow().isoformat(),
                "status": "failed",
                "error": error_msg[:1000],
            })
            await self._storage.update_schedule_status(schedule_id, ScheduleStatus.FAILED.value)
