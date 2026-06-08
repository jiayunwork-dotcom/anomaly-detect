from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from app.storage.database import StorageManager

from .models import ABTestConfig, ABTestStatus, ModelStatus, ModelVersionInfo

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, storage: StorageManager) -> None:
        self._storage = storage
        self._active_cache: dict[str, str] = {}
        self._ab_test_cache: dict[str, ABTestConfig] = {}

    async def register_model(
        self,
        name: str,
        algorithm_type: str,
        training_params: dict | None = None,
        training_data_start: str = "",
        training_data_end: str = "",
    ) -> ModelVersionInfo:
        model_id = uuid4().hex[:12]
        now = datetime.utcnow().isoformat()
        existing = await self._storage.list_model_versions(name)
        version_num = len(existing) + 1
        major = version_num
        version = f"{major}.0.0"

        model = ModelVersionInfo(
            id=model_id,
            name=name,
            algorithm_type=algorithm_type,
            version=version,
            training_params=training_params or {},
            training_data_start=training_data_start,
            training_data_end=training_data_end,
            status=ModelStatus.TRAINING,
            created_at=now,
            updated_at=now,
        )

        await self._storage.save_model_version({
            "id": model.id,
            "name": model.name,
            "algorithm_type": model.algorithm_type,
            "version": model.version,
            "training_params": json.dumps(model.training_params),
            "training_data_start": model.training_data_start,
            "training_data_end": model.training_data_end,
            "precision": model.precision,
            "recall": model.recall,
            "f1": model.f1,
            "status": model.status.value,
            "parent_version_id": model.parent_version_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        })

        logger.info("Registered model %s version %s", name, version)
        return model

    async def list_models(self) -> list[dict]:
        return await self._storage.list_model_groups()

    async def list_model_versions(self, name: str) -> list[ModelVersionInfo]:
        rows = await self._storage.list_model_versions(name)
        return [self._row_to_model(r) for r in rows]

    async def get_model(self, model_id: str) -> Optional[ModelVersionInfo]:
        row = await self._storage.get_model_version(model_id)
        if row is None:
            return None
        return self._row_to_model(row)

    async def activate_model(self, model_id: str) -> Optional[ModelVersionInfo]:
        model = await self.get_model(model_id)
        if model is None:
            return None
        if model.status == ModelStatus.ACTIVE:
            return model
        if model.status not in (ModelStatus.TRAINING, ModelStatus.RETIRED):
            return None

        ab_test = await self.get_ab_test(model.name)
        if ab_test is not None and ab_test.status == ABTestStatus.RUNNING:
            if model_id in (ab_test.primary_model_id, ab_test.challenger_model_id):
                logger.warning("Cannot activate model %s during A/B test for %s", model_id, model.name)
                return None

        current_active = await self._storage.get_active_model_version(model.name)
        if current_active:
            await self._storage.update_model_version_status(
                current_active["id"], ModelStatus.RETIRED.value
            )
            self._active_cache.pop(model.name, None)

        await self._storage.update_model_version_status(
            model_id, ModelStatus.ACTIVE.value
        )
        self._active_cache[model.name] = model_id
        logger.info("Activated model %s version %s", model.name, model.version)
        return await self.get_model(model_id)

    async def retire_model(self, model_id: str) -> Optional[ModelVersionInfo]:
        model = await self.get_model(model_id)
        if model is None:
            return None
        if model.status != ModelStatus.ACTIVE:
            return None

        ab_test = await self.get_ab_test(model.name)
        if ab_test is not None and ab_test.status == ABTestStatus.RUNNING:
            if model_id in (ab_test.primary_model_id, ab_test.challenger_model_id):
                logger.warning("Cannot retire model %s during A/B test for %s", model_id, model.name)
                return None

        await self._storage.update_model_version_status(
            model_id, ModelStatus.RETIRED.value
        )
        self._active_cache.pop(model.name, None)
        logger.info("Retired model %s version %s", model.name, model.version)
        return await self.get_model(model_id)

    async def delete_model(self, model_id: str) -> bool:
        model = await self.get_model(model_id)
        if model is None:
            return False
        if model.status != ModelStatus.RETIRED:
            return False
        await self._storage.delete_model_version(model_id)
        logger.info("Deleted model %s version %s", model.name, model.version)
        return True

    async def update_model_metrics(
        self,
        model_id: str,
        precision: float,
        recall: float,
        f1: float,
    ) -> Optional[ModelVersionInfo]:
        model = await self.get_model(model_id)
        if model is None:
            return None
        now = datetime.utcnow().isoformat()
        await self._storage.update_model_version_metrics(
            model_id, precision, recall, f1, now
        )
        return await self.get_model(model_id)

    async def update_model_status(
        self, model_id: str, status: ModelStatus
    ) -> Optional[ModelVersionInfo]:
        await self._storage.update_model_version_status(model_id, status.value)
        return await self.get_model(model_id)

    async def get_active_model(self, name: str) -> Optional[ModelVersionInfo]:
        if name in self._active_cache:
            model = await self.get_model(self._active_cache[name])
            if model and model.status == ModelStatus.ACTIVE:
                return model
        row = await self._storage.get_active_model_version(name)
        if row is None:
            return None
        self._active_cache[name] = row["id"]
        return self._row_to_model(row)

    async def load_active_cache(self) -> None:
        groups = await self._storage.list_model_groups()
        for g in groups:
            name = g.get("name", "")
            active_id = g.get("active_model_id")
            if active_id:
                self._active_cache[name] = active_id

    def _row_to_model(self, row: dict) -> ModelVersionInfo:
        params = row.get("training_params", "{}")
        if isinstance(params, str):
            params = json.loads(params)
        return ModelVersionInfo(
            id=row["id"],
            name=row["name"],
            algorithm_type=row["algorithm_type"],
            version=row["version"],
            training_params=params,
            training_data_start=row.get("training_data_start", ""),
            training_data_end=row.get("training_data_end", ""),
            precision=row.get("precision", 0.0),
            recall=row.get("recall", 0.0),
            f1=row.get("f1", 0.0),
            status=ModelStatus(row.get("status", "training")),
            parent_version_id=row.get("parent_version_id"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    async def start_ab_test(
        self,
        model_name: str,
        primary_model_id: str,
        challenger_model_id: str,
        primary_traffic_pct: float = 80.0,
        min_windows: int = 5,
        f1_improvement_threshold: float = 0.05,
    ) -> Optional[ABTestConfig]:
        existing = await self.get_ab_test(model_name)
        if existing is not None and existing.status == ABTestStatus.RUNNING:
            logger.warning("A/B test already running for %s", model_name)
            return None

        primary = await self.get_model(primary_model_id)
        challenger = await self.get_model(challenger_model_id)
        if primary is None or challenger is None:
            logger.error("Primary or challenger model not found")
            return None
        if primary.name != model_name or challenger.name != model_name:
            logger.error("Model name mismatch for A/B test")
            return None

        now = datetime.utcnow().isoformat()
        config = ABTestConfig(
            model_name=model_name,
            primary_model_id=primary_model_id,
            challenger_model_id=challenger_model_id,
            primary_traffic_pct=primary_traffic_pct,
            min_windows=min_windows,
            f1_improvement_threshold=f1_improvement_threshold,
            status=ABTestStatus.RUNNING,
            windows_completed=0,
            primary_tp=0,
            primary_fp=0,
            primary_fn=0,
            primary_precision=0.0,
            primary_recall=0.0,
            primary_f1=0.0,
            challenger_tp=0,
            challenger_fp=0,
            challenger_fn=0,
            challenger_precision=0.0,
            challenger_recall=0.0,
            challenger_f1=0.0,
            created_at=now,
            updated_at=now,
        )

        await self._storage.save_ab_test(config.model_dump())
        self._ab_test_cache[model_name] = config
        logger.info("Started A/B test for %s: primary=%s challenger=%s", model_name, primary_model_id, challenger_model_id)
        return config

    async def get_ab_test(self, model_name: str) -> Optional[ABTestConfig]:
        if model_name in self._ab_test_cache:
            cached = self._ab_test_cache[model_name]
            if cached.status == ABTestStatus.RUNNING:
                return cached

        row = await self._storage.get_ab_test(model_name)
        if row is None:
            return None
        config = self._row_to_ab_test(row)
        self._ab_test_cache[model_name] = config
        return config

    async def list_ab_tests(self) -> list[ABTestConfig]:
        rows = await self._storage.list_ab_tests()
        return [self._row_to_ab_test(row) for row in rows]

    def _row_to_ab_test(self, row: dict) -> ABTestConfig:
        return ABTestConfig(
            model_name=row["model_name"],
            primary_model_id=row["primary_model_id"],
            challenger_model_id=row["challenger_model_id"],
            primary_traffic_pct=row["primary_traffic_pct"],
            min_windows=row["min_windows"],
            f1_improvement_threshold=row["f1_improvement_threshold"],
            status=ABTestStatus(row["status"]),
            windows_completed=row["windows_completed"],
            primary_tp=row.get("primary_tp", 0),
            primary_fp=row.get("primary_fp", 0),
            primary_fn=row.get("primary_fn", 0),
            primary_precision=row["primary_precision"],
            primary_recall=row["primary_recall"],
            primary_f1=row["primary_f1"],
            challenger_tp=row.get("challenger_tp", 0),
            challenger_fp=row.get("challenger_fp", 0),
            challenger_fn=row.get("challenger_fn", 0),
            challenger_precision=row["challenger_precision"],
            challenger_recall=row["challenger_recall"],
            challenger_f1=row["challenger_f1"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            ended_at=row["ended_at"],
        )

    async def route_ab_test(self, model_name: str) -> Optional[str]:
        ab_test = await self.get_ab_test(model_name)
        if ab_test is None or ab_test.status != ABTestStatus.RUNNING:
            active = await self.get_active_model(model_name)
            return active.id if active else None

        roll = random.random() * 100
        if roll < ab_test.primary_traffic_pct:
            return ab_test.primary_model_id
        else:
            return ab_test.challenger_model_id

    async def record_ab_test_window(
        self,
        model_name: str,
        is_primary: bool,
        tp: int,
        fp: int,
        fn: int,
        ws_callback=None,
    ) -> Optional[ABTestConfig]:
        ab_test = await self.get_ab_test(model_name)
        if ab_test is None or ab_test.status != ABTestStatus.RUNNING:
            return None

        ab_test.windows_completed += 1

        if is_primary:
            ab_test.primary_tp += tp
            ab_test.primary_fp += fp
            ab_test.primary_fn += fn
            total_tp = ab_test.primary_tp
            total_fp = ab_test.primary_fp
            total_fn = ab_test.primary_fn
            ab_test.primary_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            ab_test.primary_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            ab_test.primary_f1 = (
                2 * ab_test.primary_precision * ab_test.primary_recall / (ab_test.primary_precision + ab_test.primary_recall)
                if (ab_test.primary_precision + ab_test.primary_recall) > 0 else 0.0
            )
        else:
            ab_test.challenger_tp += tp
            ab_test.challenger_fp += fp
            ab_test.challenger_fn += fn
            total_tp = ab_test.challenger_tp
            total_fp = ab_test.challenger_fp
            total_fn = ab_test.challenger_fn
            ab_test.challenger_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            ab_test.challenger_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            ab_test.challenger_f1 = (
                2 * ab_test.challenger_precision * ab_test.challenger_recall / (ab_test.challenger_precision + ab_test.challenger_recall)
                if (ab_test.challenger_precision + ab_test.challenger_recall) > 0 else 0.0
            )

        now = datetime.utcnow().isoformat()
        ab_test.updated_at = now

        if ab_test.windows_completed >= ab_test.min_windows:
            f1_diff = ab_test.challenger_f1 - ab_test.primary_f1
            if f1_diff > ab_test.f1_improvement_threshold:
                ab_test.status = ABTestStatus.COMPLETED_PROMOTED
                ab_test.ended_at = now
                await self._storage.save_ab_test(ab_test.model_dump())
                self._ab_test_cache[model_name] = ab_test

                await self._storage.update_model_version_status(
                    ab_test.primary_model_id, ModelStatus.RETIRED.value
                )
                self._active_cache.pop(model_name, None)
                await self.activate_model(ab_test.challenger_model_id)

                logger.info(
                    "A/B test completed: challenger promoted for %s (f1 diff=%.4f)",
                    model_name, f1_diff,
                )

                if ws_callback:
                    try:
                        await ws_callback({
                            "type": "ab_test_completed",
                            "model_name": model_name,
                            "result": "challenger_promoted",
                            "primary_f1": ab_test.primary_f1,
                            "challenger_f1": ab_test.challenger_f1,
                            "f1_diff": f1_diff,
                        })
                    except Exception:
                        pass

                return ab_test
            else:
                ab_test.status = ABTestStatus.COMPLETED_RETIRED
                ab_test.ended_at = now
                await self._storage.save_ab_test(ab_test.model_dump())
                self._ab_test_cache[model_name] = ab_test

                await self._storage.update_model_version_status(
                    ab_test.challenger_model_id, ModelStatus.RETIRED.value
                )

                logger.info(
                    "A/B test completed: challenger retired for %s (f1 diff=%.4f)",
                    model_name, f1_diff,
                )

                if ws_callback:
                    try:
                        await ws_callback({
                            "type": "ab_test_completed",
                            "model_name": model_name,
                            "result": "challenger_retired",
                            "primary_f1": ab_test.primary_f1,
                            "challenger_f1": ab_test.challenger_f1,
                            "f1_diff": f1_diff,
                        })
                    except Exception:
                        pass

                return ab_test

        await self._storage.save_ab_test(ab_test.model_dump())
        self._ab_test_cache[model_name] = ab_test
        return ab_test

    def is_ab_test_running(self, model_name: str) -> bool:
        ab_test = self._ab_test_cache.get(model_name)
        return ab_test is not None and ab_test.status == ABTestStatus.RUNNING
