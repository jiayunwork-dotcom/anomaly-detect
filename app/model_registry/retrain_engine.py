from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
from scipy.stats import entropy

from app.storage.database import StorageManager

from .models import ModelStatus, RetrainStrategyConfig, TriggerType
from .registry import ModelRegistry
from .training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class RetrainEngine:
    def __init__(
        self,
        storage: StorageManager,
        registry: ModelRegistry,
        pipeline: TrainingPipeline,
        ws_callback: Optional[Callable] = None,
    ) -> None:
        self._storage = storage
        self._registry = registry
        self._pipeline = pipeline
        self._ws_callback = ws_callback
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._training_locks: set[str] = set()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._check_task = asyncio.create_task(self._periodic_check())
        logger.info("RetrainEngine started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._check_task is not None:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("RetrainEngine stopped")

    async def _periodic_check(self) -> None:
        while self._running:
            try:
                await self._check_all_strategies()
            except Exception as e:
                logger.error("Periodic retrain check failed: %s", e)
            await asyncio.sleep(60)

    async def _check_all_strategies(self) -> None:
        configs = await self._storage.list_retrain_configs()
        for cfg_dict in configs:
            if not cfg_dict.get("enabled", True):
                continue
            try:
                model_name = cfg_dict["model_name"]
                active_model = await self._registry.get_active_model(model_name)
                if active_model is None:
                    continue

                config = RetrainStrategyConfig(
                    model_name=model_name,
                    trigger_type=TriggerType(cfg_dict["trigger_type"]),
                    scheduled_interval_hours=cfg_dict.get("scheduled_interval_hours", 24),
                    performance_window_size=cfg_dict.get("performance_window_size", 10),
                    performance_f1_threshold=cfg_dict.get("performance_f1_threshold", 0.7),
                    drift_kl_threshold=cfg_dict.get("drift_kl_threshold", 0.5),
                    training_data_days=cfg_dict.get("training_data_days", 30),
                    enabled=cfg_dict.get("enabled", True),
                )
                should_trigger = await self._should_trigger(config)
                if should_trigger:
                    await self.trigger_retrain(active_model.id, config.trigger_type.value)
            except Exception as e:
                logger.error("Strategy check error for %s: %s", cfg_dict.get("model_name"), e)

    async def _should_trigger(self, config: RetrainStrategyConfig) -> bool:
        model = await self._registry.get_active_model(config.model_name)
        if model is None:
            return False
        if model.status != ModelStatus.ACTIVE:
            return False
        if model.id in self._training_locks:
            return False

        if config.trigger_type == TriggerType.SCHEDULED:
            return await self._check_scheduled_trigger(config, model.id)
        elif config.trigger_type == TriggerType.PERFORMANCE:
            return await self._check_performance_trigger(config, model.id)
        elif config.trigger_type == TriggerType.DATA_DRIFT:
            return await self._check_drift_trigger(config, model)
        return False

    async def _check_scheduled_trigger(self, config: RetrainStrategyConfig, active_model_id: str) -> bool:
        last_training = await self._storage.get_last_training_time(active_model_id)
        if last_training is None:
            return True
        last_dt = datetime.fromisoformat(last_training)
        threshold = timedelta(hours=config.scheduled_interval_hours)
        return datetime.utcnow() - last_dt >= threshold

    async def _check_performance_trigger(self, config: RetrainStrategyConfig, active_model_id: str) -> bool:
        recent_f1s = await self._storage.get_recent_f1_scores(
            active_model_id, config.performance_window_size
        )
        if len(recent_f1s) < config.performance_window_size:
            return False
        avg_f1 = sum(recent_f1s) / len(recent_f1s)
        return avg_f1 < config.performance_f1_threshold

    async def _check_drift_trigger(self, config: RetrainStrategyConfig, model: "ModelVersionInfo") -> bool:
        training_data_start = model.training_data_start
        training_data_end = model.training_data_end
        if not training_data_start or not training_data_end:
            return False

        try:
            train_start = datetime.fromisoformat(training_data_start)
            train_end = datetime.fromisoformat(training_data_end)
            now = datetime.utcnow()
            recent_start = now - timedelta(days=1)

            metrics_list = await self._storage.list_metrics()
            train_samples: list[float] = []
            recent_samples: list[float] = []

            for m in metrics_list:
                train_df = await self._storage.load_metric_data(m.name, train_start, train_end)
                recent_df = await self._storage.load_metric_data(m.name, recent_start, now)
                if not train_df.empty and "value" in train_df.columns:
                    train_samples.extend(train_df["value"].dropna().tolist())
                if not recent_df.empty and "value" in recent_df.columns:
                    recent_samples.extend(recent_df["value"].dropna().tolist())

            if len(train_samples) < 50 or len(recent_samples) < 50:
                return False

            train_arr = np.array(train_samples[:1000])
            recent_arr = np.array(recent_samples[:1000])

            bins = np.histogram_bin_edges(np.concatenate([train_arr, recent_arr]), bins=50)
            train_hist, _ = np.histogram(train_arr, bins=bins, density=True)
            recent_hist, _ = np.histogram(recent_arr, bins=bins, density=True)

            train_hist = train_hist + 1e-10
            recent_hist = recent_hist + 1e-10

            kl = entropy(recent_hist, train_hist)
            return float(kl) > config.drift_kl_threshold

        except Exception as e:
            logger.error("Drift detection error: %s", e)
            return False

    async def trigger_retrain(
        self, model_id: str, trigger_reason: str = "manual"
    ) -> Optional[str]:
        if model_id in self._training_locks:
            logger.info("Training already in progress for %s, skipping", model_id)
            return None

        model = await self._registry.get_model(model_id)
        if model is None:
            logger.error("Model %s not found for retraining", model_id)
            return None

        self._training_locks.add(model_id)

        try:
            retrain_config = await self._storage.get_retrain_config(model.name)
            training_days = retrain_config.get("training_data_days", 30) if retrain_config else 30

            if trigger_reason == "initial":
                context = await self._pipeline.run_training(
                    new_model_id=model_id,
                    parent_model_id=None,
                    training_data_days=training_days,
                    trigger_reason=trigger_reason,
                )

                if context.error is not None:
                    await self._registry.update_model_status(model_id, ModelStatus.FAILED)
                    logger.error("Initial training failed for %s: %s", model_id, context.error)
                else:
                    await self._registry.activate_model(model_id)
                    context.auto_activated = True
                    logger.info("Initial training completed, activated model %s", model_id)

                await self._storage.save_training_context(context.model_dump())

                if self._ws_callback:
                    try:
                        await self._ws_callback({
                            "type": "retrain_complete",
                            "model_id": model_id,
                            "model_name": model.name,
                            "version": model.version,
                            "auto_activated": context.auto_activated,
                            "new_f1": context.new_f1,
                            "old_f1": context.old_f1,
                            "trigger_reason": trigger_reason,
                        })
                    except Exception:
                        pass

                return model_id

            new_model = await self._registry.register_model(
                name=model.name,
                algorithm_type=model.algorithm_type,
                training_params=model.training_params,
                training_data_start=(datetime.utcnow() - timedelta(days=training_days)).isoformat(),
                training_data_end=datetime.utcnow().isoformat(),
            )

            await self._storage.update_model_parent(new_model.id, model_id)

            context = await self._pipeline.run_training(
                new_model_id=new_model.id,
                parent_model_id=model_id,
                training_data_days=training_days,
                trigger_reason=trigger_reason,
            )

            if context.error is not None:
                await self._registry.update_model_status(new_model.id, ModelStatus.FAILED)
                logger.error("Retraining failed for %s: %s", new_model.id, context.error)
            else:
                if context.new_f1 > context.old_f1:
                    await self._registry.activate_model(new_model.id)
                    context.auto_activated = True
                    logger.info(
                        "Auto-activated new model %s (f1=%.4f > %.4f)",
                        new_model.id, context.new_f1, context.old_f1,
                    )
                else:
                    await self._registry.update_model_status(new_model.id, ModelStatus.FAILED)
                    logger.info(
                        "New model %s f1=%.4f <= %.4f, keeping current",
                        new_model.id, context.new_f1, context.old_f1,
                    )

                await self._storage.save_training_context(context.model_dump())

            if self._ws_callback:
                try:
                    await self._ws_callback({
                        "type": "retrain_complete",
                        "model_id": new_model.id,
                        "model_name": model.name,
                        "version": new_model.version,
                        "auto_activated": context.auto_activated,
                        "new_f1": context.new_f1,
                        "old_f1": context.old_f1,
                        "trigger_reason": trigger_reason,
                    })
                except Exception:
                    pass

            return new_model.id

        except Exception as e:
            logger.error("Retrain error for %s: %s", model_id, e)
            traceback.print_exc()
            return None
        finally:
            self._training_locks.discard(model_id)

    async def save_retrain_config(self, config: RetrainStrategyConfig) -> None:
        await self._storage.save_retrain_config({
            "model_name": config.model_name,
            "trigger_type": config.trigger_type.value,
            "scheduled_interval_hours": config.scheduled_interval_hours,
            "performance_window_size": config.performance_window_size,
            "performance_f1_threshold": config.performance_f1_threshold,
            "drift_kl_threshold": config.drift_kl_threshold,
            "training_data_days": config.training_data_days,
            "enabled": config.enabled,
        })

    async def get_retrain_config(self, model_name: str) -> Optional[RetrainStrategyConfig]:
        cfg = await self._storage.get_retrain_config(model_name)
        if cfg is None:
            return None
        return RetrainStrategyConfig(
            model_name=cfg["model_name"],
            trigger_type=TriggerType(cfg["trigger_type"]),
            scheduled_interval_hours=cfg.get("scheduled_interval_hours", 24),
            performance_window_size=cfg.get("performance_window_size", 10),
            performance_f1_threshold=cfg.get("performance_f1_threshold", 0.7),
            drift_kl_threshold=cfg.get("drift_kl_threshold", 0.5),
            training_data_days=cfg.get("training_data_days", 30),
            enabled=cfg.get("enabled", True),
        )

    def is_training_in_progress(self, model_id: str) -> bool:
        return model_id in self._training_locks
