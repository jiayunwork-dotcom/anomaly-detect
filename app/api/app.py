from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.ingestion.parser import parse_csv, parse_influxdb_lp, parse_prometheus_json
from app.labeling.loop import LabelingLoop
from app.model_registry import ModelRegistry, RetrainEngine, TrainingPipeline
from app.model_registry.models import ABTestStatus, ModelStatus, RetrainStrategyConfig, TriggerType
from app.scheduler.engine import SchedulerEngine
from app.storage.database import StorageManager

from .models import (
    ABTestResponse,
    ABTestStartRequest,
    AnomalyEventResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    DataFormat,
    DataImportRequest,
    DataImportResponse,
    DetectionRequest,
    DetectionResponse,
    DetectionResult,
    LabelRequest,
    LabelType,
    ModelAlertResponse,
    ModelCompareRequest,
    ModelCompareResponse,
    ModelGroupResponse,
    ModelRegisterRequest,
    ModelVersionResponse,
    RetrainConfigRequest,
    RetrainConfigResponse,
    ScheduleCreateRequest,
    ScheduleResponse,
    TaskStatus,
    TrainingContextResponse,
    TrainingProgressResponse,
)
from .tasks import get_task, register_task, run_batch_detection, run_detection_task

storage: StorageManager = StorageManager()
_labeling_loop: Optional[LabelingLoop] = None
_scheduler_engine: Optional[SchedulerEngine] = None
_model_registry: Optional[ModelRegistry] = None
_retrain_engine: Optional[RetrainEngine] = None
_training_pipeline: Optional[TrainingPipeline] = None
_ws_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _labeling_loop, _scheduler_engine, _model_registry, _retrain_engine, _training_pipeline
    await storage.init()
    _labeling_loop = LabelingLoop(storage)
    _scheduler_engine = SchedulerEngine(storage, _ws_broadcast)
    _model_registry = ModelRegistry(storage)
    await _model_registry.load_active_cache()
    _training_pipeline = TrainingPipeline(storage, _model_registry, _ws_broadcast)
    _retrain_engine = RetrainEngine(storage, _model_registry, _training_pipeline, _ws_broadcast)
    await _scheduler_engine.start()
    await _retrain_engine.start()
    yield
    await _retrain_engine.stop()
    await _scheduler_engine.stop()
    await storage.close()


app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)


async def _ws_broadcast(data: dict) -> None:
    disconnected: list[WebSocket] = []
    for ws in _ws_connections:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_connections.remove(ws)


@app.post("/api/detect", response_model=DetectionResponse)
async def submit_detection(request: DetectionRequest):
    task_id = uuid4().hex
    register_task(task_id)
    asyncio.create_task(run_detection_task(task_id, request, storage, _ws_broadcast, _model_registry))
    return DetectionResponse(task_id=task_id, status=TaskStatus.PENDING)


@app.get("/api/tasks/{task_id}", response_model=DetectionResult)
async def get_task_status(task_id: str):
    state = get_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if state.result is not None:
        return state.result
    return DetectionResult(task_id=task_id, status=state.status)


@app.post("/api/detect/batch", response_model=BatchDetectionResponse)
async def submit_batch_detection(request: BatchDetectionRequest):
    task_ids = [uuid4().hex for _ in request.items]
    for tid in task_ids:
        register_task(tid)
    asyncio.create_task(
        run_batch_detection(task_ids, request.items, storage, _ws_broadcast, _model_registry)
    )
    return BatchDetectionResponse(task_ids=task_ids)


@app.websocket("/ws/anomalies")
async def websocket_anomalies(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


@app.post("/api/labels")
async def submit_label(request: LabelRequest):
    if _labeling_loop is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if request.label == LabelType.TP:
        await _labeling_loop.confirm_event(request.event_id)
    else:
        await _labeling_loop.mark_false_alarm(request.event_id)
    return {"status": "ok", "event_id": request.event_id, "label": request.label.value}


@app.get("/api/anomalies", response_model=list[AnomalyEventResponse])
async def list_anomalies(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    metric_name: Optional[str] = None,
):
    start = datetime.fromisoformat(start_time) if start_time else datetime(2000, 1, 1)
    end = datetime.fromisoformat(end_time) if end_time else datetime.utcnow()
    events = await storage.get_anomaly_events(start, end, metric_name)
    return [AnomalyEventResponse(**e) for e in events]


@app.get("/api/metrics")
async def list_metrics():
    metrics = await storage.list_metrics()
    return [
        {
            "name": m.name,
            "unit": m.unit,
            "source": m.source,
            "frequency": m.frequency,
            "valid_min": m.valid_min,
            "valid_max": m.valid_max,
            "created_at": m.created_at,
        }
        for m in metrics
    ]


@app.get("/api/algorithms/performance")
async def algorithm_performance():
    if _labeling_loop is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await _labeling_loop.get_algorithm_stats()


@app.post("/api/data/import", response_model=DataImportResponse)
async def import_data(request: DataImportRequest):
    try:
        if request.format == DataFormat.CSV:
            df, metadata = parse_csv(request.data)
        elif request.format == DataFormat.PROMETHEUS:
            df, metadata = parse_prometheus_json(request.data)
        elif request.format == DataFormat.INFLUXDB:
            df, metadata = parse_influxdb_lp(request.data)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    metric_names: list[str] = []
    for meta in metadata:
        col_name = meta["name"]
        if col_name in df.columns:
            await storage.save_metric_data(col_name, df[[col_name]])
            await storage.save_metric_metadata(meta)
            metric_names.append(col_name)

    return DataImportResponse(
        metrics=metric_names,
        rows_imported=len(df),
    )


@app.post("/api/schedules", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleCreateRequest):
    if _scheduler_engine is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    schedule_id = await _scheduler_engine.create_schedule(request)
    schedule = _scheduler_engine.get_schedule(schedule_id)
    if schedule is None:
        raise HTTPException(status_code=500, detail="Failed to create schedule")
    return ScheduleResponse(**schedule)


@app.get("/api/schedules", response_model=list[ScheduleResponse])
async def list_schedules():
    if _scheduler_engine is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    schedules = _scheduler_engine.list_schedules()
    return [ScheduleResponse(**s) for s in schedules]


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    if _scheduler_engine is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    success = await _scheduler_engine.delete_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "ok", "schedule_id": schedule_id}


@app.put("/api/schedules/{schedule_id}/pause")
async def pause_schedule(schedule_id: str):
    if _scheduler_engine is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    success = await _scheduler_engine.pause_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=400, detail="Schedule not found or not running")
    return {"status": "ok", "schedule_id": schedule_id, "state": "paused"}


@app.put("/api/schedules/{schedule_id}/resume")
async def resume_schedule(schedule_id: str):
    if _scheduler_engine is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    success = await _scheduler_engine.resume_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=400, detail="Schedule not found or not paused")
    return {"status": "ok", "schedule_id": schedule_id, "state": "running"}


@app.get("/api/schedules/{schedule_id}/history")
async def get_schedule_history(schedule_id: str, limit: int = 20):
    history = await storage.get_execution_history(schedule_id, limit)
    return history


@app.post("/api/models", response_model=ModelVersionResponse)
async def register_model(request: ModelRegisterRequest):
    if _model_registry is None or _retrain_engine is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    model = await _model_registry.register_model(
        name=request.name,
        algorithm_type=request.algorithm_type,
        training_params=request.training_params,
        training_data_start=request.training_data_start,
        training_data_end=request.training_data_end,
    )
    asyncio.create_task(_retrain_engine.trigger_retrain(model.id, "initial"))
    return ModelVersionResponse(**model.model_dump())


@app.get("/api/models", response_model=list[ModelGroupResponse])
async def list_models():
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    groups = await _model_registry.list_models()
    return [ModelGroupResponse(**g) for g in groups]


@app.get("/api/models/{name}/versions", response_model=list[ModelVersionResponse])
async def list_model_versions(name: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    versions = await _model_registry.list_model_versions(name)
    return [ModelVersionResponse(**v.model_dump()) for v in versions]


@app.get("/api/models/{model_id}/detail", response_model=ModelVersionResponse)
async def get_model_detail(model_id: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    model = await _model_registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelVersionResponse(**model.model_dump())


@app.put("/api/models/{model_id}/activate", response_model=ModelVersionResponse)
async def activate_model(model_id: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    model = await _model_registry.activate_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found or cannot be activated")
    return ModelVersionResponse(**model.model_dump())


@app.put("/api/models/{model_id}/retire", response_model=ModelVersionResponse)
async def retire_model(model_id: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    model = await _model_registry.retire_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found or not active")
    return ModelVersionResponse(**model.model_dump())


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    success = await _model_registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Model not found or not retired")
    return {"status": "ok", "model_id": model_id}


@app.post("/api/models/{model_id}/retrain")
async def trigger_retrain(model_id: str):
    if _retrain_engine is None:
        raise HTTPException(status_code=503, detail="Retrain engine not initialized")
    model = await _model_registry.get_model(model_id) if _model_registry else None
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if _retrain_engine.is_training_in_progress(model.name):
        raise HTTPException(status_code=409, detail="Training already in progress for this algorithm")
    new_model_id = await _retrain_engine.trigger_retrain(model_id, "manual")
    if new_model_id is None:
        raise HTTPException(status_code=400, detail="Failed to trigger retraining")
    return {"status": "ok", "new_model_id": new_model_id, "trigger": "manual"}


@app.get("/api/models/{model_id}/progress", response_model=TrainingProgressResponse)
async def get_training_progress(model_id: str):
    if _training_pipeline is None:
        raise HTTPException(status_code=503, detail="Training pipeline not initialized")
    progress = _training_pipeline.get_progress(model_id)
    if progress is None:
        return TrainingProgressResponse(model_id=model_id, stage="idle")
    return TrainingProgressResponse(**progress.model_dump())


@app.get("/api/models/{model_id}/training-history", response_model=list[TrainingContextResponse])
async def get_training_history(model_id: str, limit: int = 20):
    contexts = await storage.get_training_contexts(model_id, limit)
    return [TrainingContextResponse(**c) for c in contexts]


@app.post("/api/models/retrain-config", response_model=RetrainConfigResponse)
async def save_retrain_config(request: RetrainConfigRequest):
    if _retrain_engine is None:
        raise HTTPException(status_code=503, detail="Retrain engine not initialized")
    config = RetrainStrategyConfig(
        model_name=request.model_name,
        trigger_type=TriggerType(request.trigger_type),
        scheduled_interval_hours=request.scheduled_interval_hours,
        performance_window_size=request.performance_window_size,
        performance_f1_threshold=request.performance_f1_threshold,
        drift_kl_threshold=request.drift_kl_threshold,
        training_data_days=request.training_data_days,
        enabled=request.enabled,
    )
    await _retrain_engine.save_retrain_config(config)
    return RetrainConfigResponse(**config.model_dump())


@app.get("/api/models/{model_name}/retrain-config", response_model=RetrainConfigResponse)
async def get_retrain_config(model_name: str):
    if _retrain_engine is None:
        raise HTTPException(status_code=503, detail="Retrain engine not initialized")
    config = await _retrain_engine.get_retrain_config(model_name)
    if config is None:
        raise HTTPException(status_code=404, detail="Retrain config not found")
    return RetrainConfigResponse(**config.model_dump())


@app.post("/api/models/compare", response_model=ModelCompareResponse)
async def compare_models(request: ModelCompareRequest):
    if _training_pipeline is None:
        raise HTTPException(status_code=503, detail="Training pipeline not initialized")
    result = await _training_pipeline.compare_models(request.model_a_id, request.model_b_id)
    if result is None:
        raise HTTPException(status_code=400, detail="Comparison failed - models not found or no test data")
    return ModelCompareResponse(**result)


@app.post("/api/models/ab-test", response_model=ABTestResponse)
async def start_ab_test(request: ABTestStartRequest):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    ab_test = await _model_registry.start_ab_test(
        model_name=request.model_name,
        primary_model_id=request.primary_model_id,
        challenger_model_id=request.challenger_model_id,
        primary_traffic_pct=request.primary_traffic_pct,
        min_windows=request.min_windows,
        f1_improvement_threshold=request.f1_improvement_threshold,
    )
    if ab_test is None:
        raise HTTPException(status_code=400, detail="Failed to start A/B test - test already running or models not found")
    return ABTestResponse(**ab_test.model_dump())


@app.get("/api/models/ab-tests", response_model=list[ABTestResponse])
async def list_ab_tests():
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    tests = await _model_registry.list_ab_tests()
    return [ABTestResponse(**t.model_dump()) for t in tests]


@app.get("/api/models/{model_name}/ab-test", response_model=ABTestResponse)
async def get_ab_test(model_name: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    ab_test = await _model_registry.get_ab_test(model_name)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found for this model")
    return ABTestResponse(**ab_test.model_dump())


@app.delete("/api/models/{model_name}/ab-test")
async def stop_ab_test(model_name: str):
    if _model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    ab_test = await _model_registry.get_ab_test(model_name)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")
    if ab_test.status != ABTestStatus.RUNNING:
        raise HTTPException(status_code=400, detail="A/B test is not running")
    await storage.delete_ab_test(model_name)
    _model_registry._ab_test_cache.pop(model_name, None)
    return {"status": "ok", "model_name": model_name}


@app.get("/api/models/alerts", response_model=list[ModelAlertResponse])
async def list_model_alerts(dismissed: Optional[bool] = None):
    alerts = await storage.list_model_alerts(dismissed=dismissed)
    return [ModelAlertResponse(**a) for a in alerts]


@app.put("/api/models/alerts/{alert_id}/dismiss", response_model=ModelAlertResponse)
async def dismiss_model_alert(alert_id: int):
    success = await storage.dismiss_model_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already dismissed")
    alerts = await storage.list_model_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            return ModelAlertResponse(**a)
    raise HTTPException(status_code=404, detail="Alert not found after dismissal")
