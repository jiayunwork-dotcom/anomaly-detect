from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from app.storage.database import StorageManager

from .models import ModelStatus, ModelVersionInfo

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, storage: StorageManager) -> None:
        self._storage = storage
        self._active_cache: dict[str, str] = {}

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
