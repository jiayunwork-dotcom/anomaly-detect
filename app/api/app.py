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
from app.storage.database import StorageManager

from .models import (
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
    TaskStatus,
)
from .tasks import get_task, register_task, run_batch_detection, run_detection_task

storage: StorageManager = StorageManager()
_labeling_loop: Optional[LabelingLoop] = None
_ws_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _labeling_loop
    await storage.init()
    _labeling_loop = LabelingLoop(storage)
    yield
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
    asyncio.create_task(run_detection_task(task_id, request, storage, _ws_broadcast))
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
        run_batch_detection(task_ids, request.items, storage, _ws_broadcast)
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
