from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiosqlite
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pydantic import BaseModel, Field


class MetricMetadata(BaseModel):
    name: str
    unit: str = ""
    source: str = ""
    frequency: float = 0.0
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AnomalyEvent(BaseModel):
    metric_name: str
    start_time: str
    end_time: str
    anomaly_type: str = ""
    severity: float = 0.0
    algorithm: str = ""
    is_confirmed: bool = False
    confirmed_as: Optional[str] = None
    pattern_id: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AlertEvent(BaseModel):
    root_metric: str
    related_metrics: str = "[]"
    start_time: str
    severity: float = 0.0
    suppressed: bool = False
    channel: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Pattern(BaseModel):
    name: str
    metrics_json: str = "[]"
    shape: str = ""
    duration_seconds: float = 0.0
    root_cause: str = ""
    resolution: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AlgorithmPerformance(BaseModel):
    algorithm_name: str
    tp_count: int = 0
    fp_count: int = 0
    precision_val: float = 0.0
    recall_val: float = 0.0
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


SCHEMA_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    unit TEXT DEFAULT '',
    source TEXT DEFAULT '',
    frequency REAL DEFAULT 0.0,
    valid_min REAL,
    valid_max REAL,
    created_at TEXT NOT NULL
)
"""

SCHEMA_ANOMALY_EVENTS = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    anomaly_type TEXT DEFAULT '',
    severity REAL DEFAULT 0.0,
    algorithm TEXT DEFAULT '',
    is_confirmed INTEGER DEFAULT 0,
    confirmed_as TEXT,
    pattern_id INTEGER,
    created_at TEXT NOT NULL
)
"""

SCHEMA_ALERT_EVENTS = """
CREATE TABLE IF NOT EXISTS alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_metric TEXT NOT NULL,
    related_metrics TEXT DEFAULT '[]',
    start_time TEXT NOT NULL,
    severity REAL DEFAULT 0.0,
    suppressed INTEGER DEFAULT 0,
    channel TEXT DEFAULT '',
    created_at TEXT NOT NULL
)
"""

SCHEMA_PATTERN_LIBRARY = """
CREATE TABLE IF NOT EXISTS pattern_library (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    metrics_json TEXT DEFAULT '[]',
    shape TEXT DEFAULT '',
    duration_seconds REAL DEFAULT 0.0,
    root_cause TEXT DEFAULT '',
    resolution TEXT DEFAULT '',
    created_at TEXT NOT NULL
)
"""

SCHEMA_ALGORITHM_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS algorithm_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm_name TEXT UNIQUE NOT NULL,
    tp_count INTEGER DEFAULT 0,
    fp_count INTEGER DEFAULT 0,
    precision_val REAL DEFAULT 0.0,
    recall_val REAL DEFAULT 0.0,
    last_updated TEXT NOT NULL
)
"""

SCHEMA_SCHEDULES = """
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Untitled',
    interval TEXT NOT NULL DEFAULT '5m',
    cron_expression TEXT,
    metrics TEXT DEFAULT '[]',
    algorithms TEXT DEFAULT '["three_sigma","iqr"]',
    ensemble_mode TEXT DEFAULT 'majority',
    weights TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running',
    last_run_time TEXT,
    last_anomaly_count INTEGER DEFAULT 0,
    last_alert_triggered INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

SCHEMA_EXECUTION_HISTORY = """
CREATE TABLE IF NOT EXISTS execution_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    anomaly_count INTEGER DEFAULT 0,
    alert_triggered INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    error TEXT,
    FOREIGN KEY (schedule_id) REFERENCES schedules(id)
)
"""

SCHEMA_MODEL_VERSIONS = """
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    algorithm_type TEXT NOT NULL,
    version TEXT NOT NULL,
    training_params TEXT DEFAULT '{}',
    training_data_start TEXT DEFAULT '',
    training_data_end TEXT DEFAULT '',
    precision REAL DEFAULT 0.0,
    recall REAL DEFAULT 0.0,
    f1 REAL DEFAULT 0.0,
    status TEXT DEFAULT 'training',
    parent_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

SCHEMA_RETRAIN_CONFIGS = """
CREATE TABLE IF NOT EXISTS retrain_configs (
    model_name TEXT PRIMARY KEY,
    trigger_type TEXT DEFAULT 'scheduled',
    scheduled_interval_hours INTEGER DEFAULT 24,
    performance_window_size INTEGER DEFAULT 10,
    performance_f1_threshold REAL DEFAULT 0.7,
    drift_kl_threshold REAL DEFAULT 0.5,
    training_data_days INTEGER DEFAULT 30,
    enabled INTEGER DEFAULT 1
)
"""

SCHEMA_TRAINING_CONTEXTS = """
CREATE TABLE IF NOT EXISTS training_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    training_data_count INTEGER DEFAULT 0,
    training_duration_seconds REAL DEFAULT 0.0,
    stages TEXT DEFAULT '[]',
    old_precision REAL DEFAULT 0.0,
    old_recall REAL DEFAULT 0.0,
    old_f1 REAL DEFAULT 0.0,
    new_precision REAL DEFAULT 0.0,
    new_recall REAL DEFAULT 0.0,
    new_f1 REAL DEFAULT 0.0,
    auto_activated INTEGER DEFAULT 0,
    error TEXT,
    completed_at TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES model_versions(id)
)
"""

SCHEMA_MODEL_F1_HISTORY = """
CREATE TABLE IF NOT EXISTS model_f1_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    f1_score REAL NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES model_versions(id)
)
"""


class StorageManager:
    def __init__(self, db_path: str = "metadata.db", data_dir: str = "data") -> None:
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute(SCHEMA_METRICS)
        await self._conn.execute(SCHEMA_ANOMALY_EVENTS)
        await self._conn.execute(SCHEMA_ALERT_EVENTS)
        await self._conn.execute(SCHEMA_PATTERN_LIBRARY)
        await self._conn.execute(SCHEMA_ALGORITHM_PERFORMANCE)
        await self._conn.execute(SCHEMA_SCHEDULES)
        await self._conn.execute(SCHEMA_EXECUTION_HISTORY)
        await self._conn.execute(SCHEMA_MODEL_VERSIONS)
        await self._conn.execute(SCHEMA_RETRAIN_CONFIGS)
        await self._conn.execute(SCHEMA_TRAINING_CONTEXTS)
        await self._conn.execute(SCHEMA_MODEL_F1_HISTORY)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save_metric_data(self, metric_name: str, df: pd.DataFrame) -> None:
        metric_dir = self.data_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

        for day, group in df.groupby(df.index.date):
            path = metric_dir / f"{day.isoformat()}.parquet"
            group_copy = group.copy()
            group_copy.index.name = "timestamp"
            table = pa.Table.from_pandas(group_copy.reset_index())
            pq.write_table(table, path)

    async def load_metric_data(
        self, metric_name: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        metric_dir = self.data_dir / metric_name
        if not metric_dir.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            path = metric_dir / f"{current_date.isoformat()}.parquet"
            if path.exists():
                table = pq.read_table(path)
                day_df = table.to_pandas()
                day_df["timestamp"] = pd.to_datetime(day_df["timestamp"])
                day_df = day_df.set_index("timestamp")
                frames.append(day_df)
            current_date += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames)
        result = result.sort_index()
        result = result[start_time:end_time]
        return result

    async def save_metric_metadata(self, metadata_dict: dict) -> None:
        meta = MetricMetadata(**metadata_dict)
        await self._conn.execute(
            """INSERT OR REPLACE INTO metrics
               (name, unit, source, frequency, valid_min, valid_max, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (meta.name, meta.unit, meta.source, meta.frequency,
             meta.valid_min, meta.valid_max, meta.created_at),
        )
        await self._conn.commit()

    async def get_metric_metadata(self, metric_name: str) -> Optional[MetricMetadata]:
        cursor = await self._conn.execute(
            "SELECT name, unit, source, frequency, valid_min, valid_max, created_at FROM metrics WHERE name = ?",
            (metric_name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return MetricMetadata(
            name=row[0], unit=row[1], source=row[2], frequency=row[3],
            valid_min=row[4], valid_max=row[5], created_at=row[6],
        )

    async def list_metrics(self) -> list[MetricMetadata]:
        cursor = await self._conn.execute(
            "SELECT name, unit, source, frequency, valid_min, valid_max, created_at FROM metrics ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [
            MetricMetadata(
                name=r[0], unit=r[1], source=r[2], frequency=r[3],
                valid_min=r[4], valid_max=r[5], created_at=r[6],
            )
            for r in rows
        ]

    async def save_anomaly_event(self, event_dict: dict) -> int:
        event = AnomalyEvent(**event_dict)
        cursor = await self._conn.execute(
            """INSERT INTO anomaly_events
               (metric_name, start_time, end_time, anomaly_type, severity, algorithm,
                is_confirmed, confirmed_as, pattern_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.metric_name, event.start_time, event.end_time, event.anomaly_type,
             event.severity, event.algorithm, int(event.is_confirmed),
             event.confirmed_as, event.pattern_id, event.created_at),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def get_anomaly_events(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_name: Optional[str] = None,
    ) -> list[dict]:
        query = (
            "SELECT id, metric_name, start_time, end_time, anomaly_type, severity, "
            "algorithm, is_confirmed, confirmed_as, pattern_id, created_at "
            "FROM anomaly_events WHERE start_time >= ? AND end_time <= ?"
        )
        params: list = [start_time.isoformat(), end_time.isoformat()]
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        query += " ORDER BY start_time"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "metric_name": r[1], "start_time": r[2], "end_time": r[3],
                "anomaly_type": r[4], "severity": r[5], "algorithm": r[6],
                "is_confirmed": bool(r[7]), "confirmed_as": r[8], "pattern_id": r[9],
                "created_at": r[10],
            }
            for r in rows
        ]

    async def save_alert_event(self, event_dict: dict) -> int:
        event = AlertEvent(**event_dict)
        cursor = await self._conn.execute(
            """INSERT INTO alert_events
               (root_metric, related_metrics, start_time, severity, suppressed, channel, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event.root_metric, event.related_metrics, event.start_time,
             event.severity, int(event.suppressed), event.channel, event.created_at),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def update_anomaly_confirmation(self, event_id: int, confirmed_as: str) -> None:
        await self._conn.execute(
            "UPDATE anomaly_events SET is_confirmed = 1, confirmed_as = ? WHERE id = ?",
            (confirmed_as, event_id),
        )
        await self._conn.commit()

    async def save_pattern(self, pattern_dict: dict) -> int:
        pattern = Pattern(**pattern_dict)
        cursor = await self._conn.execute(
            """INSERT INTO pattern_library
               (name, metrics_json, shape, duration_seconds, root_cause, resolution, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pattern.name, pattern.metrics_json, pattern.shape,
             pattern.duration_seconds, pattern.root_cause, pattern.resolution,
             pattern.created_at),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def search_similar_pattern(
        self,
        metrics: list[str],
        shape: Optional[str] = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT id, name, metrics_json, shape, duration_seconds, root_cause, resolution, created_at "
            "FROM pattern_library"
        )
        rows = await cursor.fetchall()
        results: list[dict] = []
        query_metrics_set = set(metrics)

        for r in rows:
            pattern_metrics = set(json.loads(r[2]))
            if not query_metrics_set and not pattern_metrics:
                jaccard = 1.0
            elif not query_metrics_set or not pattern_metrics:
                jaccard = 0.0
            else:
                intersection = query_metrics_set & pattern_metrics
                union = query_metrics_set | pattern_metrics
                jaccard = len(intersection) / len(union)

            similarity = jaccard
            if shape is not None and r[3]:
                shape_match = 1.0 if shape == r[3] else 0.0
                similarity = jaccard * 0.7 + shape_match * 0.3

            if similarity >= threshold:
                results.append({
                    "id": r[0], "name": r[1], "metrics_json": r[2], "shape": r[3],
                    "duration_seconds": r[4], "root_cause": r[5], "resolution": r[6],
                    "created_at": r[7], "similarity": similarity,
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    async def update_algorithm_performance(
        self, algorithm_name: str, tp: int, fp: int
    ) -> None:
        cursor = await self._conn.execute(
            "SELECT tp_count, fp_count FROM algorithm_performance WHERE algorithm_name = ?",
            (algorithm_name,),
        )
        row = await cursor.fetchone()
        now = datetime.utcnow().isoformat()

        if row:
            new_tp = row[0] + tp
            new_fp = row[1] + fp
            total = new_tp + new_fp
            precision_val = new_tp / total if total > 0 else 0.0
            await self._conn.execute(
                """UPDATE algorithm_performance
                   SET tp_count = ?, fp_count = ?, precision_val = ?, last_updated = ?
                   WHERE algorithm_name = ?""",
                (new_tp, new_fp, precision_val, now, algorithm_name),
            )
        else:
            total = tp + fp
            precision_val = tp / total if total > 0 else 0.0
            await self._conn.execute(
                """INSERT INTO algorithm_performance
                   (algorithm_name, tp_count, fp_count, precision_val, recall_val, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (algorithm_name, tp, fp, precision_val, 0.0, now),
            )
        await self._conn.commit()

    async def get_algorithm_performance(self) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT id, algorithm_name, tp_count, fp_count, precision_val, recall_val, last_updated "
            "FROM algorithm_performance"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "algorithm_name": r[1], "tp_count": r[2], "fp_count": r[3],
                "precision_val": r[4], "recall_val": r[5], "last_updated": r[6],
            }
            for r in rows
        ]

    async def export_labels_csv(
        self, start_time: datetime, end_time: datetime, output_path: str
    ) -> None:
        events = await self.get_anomaly_events(start_time, end_time)
        if not events:
            pd.DataFrame().to_csv(output_path, index=False)
        else:
            pd.DataFrame(events).to_csv(output_path, index=False)

    async def save_schedule(self, schedule_dict: dict) -> None:
        await self._conn.execute(
            """INSERT INTO schedules
               (id, name, interval, cron_expression, metrics, algorithms,
                ensemble_mode, weights, status, last_run_time,
                last_anomaly_count, last_alert_triggered, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                schedule_dict["id"],
                schedule_dict.get("name", "Untitled"),
                schedule_dict.get("interval", "5m"),
                schedule_dict.get("cron_expression"),
                schedule_dict.get("metrics", "[]"),
                schedule_dict.get("algorithms", '["three_sigma","iqr"]'),
                schedule_dict.get("ensemble_mode", "majority"),
                schedule_dict.get("weights", "{}"),
                schedule_dict.get("status", "running"),
                schedule_dict.get("last_run_time"),
                schedule_dict.get("last_anomaly_count", 0),
                schedule_dict.get("last_alert_triggered", 0),
                schedule_dict.get("created_at", datetime.utcnow().isoformat()),
            ),
        )
        await self._conn.commit()

    async def get_all_schedules(self) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT id, name, interval, cron_expression, metrics, algorithms, "
            "ensemble_mode, weights, status, last_run_time, "
            "last_anomaly_count, last_alert_triggered, created_at "
            "FROM schedules ORDER BY created_at"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "name": r[1], "interval": r[2],
                "cron_expression": r[3], "metrics": r[4], "algorithms": r[5],
                "ensemble_mode": r[6], "weights": r[7], "status": r[8],
                "last_run_time": r[9], "last_anomaly_count": r[10],
                "last_alert_triggered": r[11], "created_at": r[12],
            }
            for r in rows
        ]

    async def delete_schedule(self, schedule_id: str) -> None:
        await self._conn.execute(
            "DELETE FROM execution_history WHERE schedule_id = ?",
            (schedule_id,),
        )
        await self._conn.execute(
            "DELETE FROM schedules WHERE id = ?",
            (schedule_id,),
        )
        await self._conn.commit()

    async def update_schedule_status(self, schedule_id: str, status: str) -> None:
        await self._conn.execute(
            "UPDATE schedules SET status = ? WHERE id = ?",
            (status, schedule_id),
        )
        await self._conn.commit()

    async def update_schedule_last_run(
        self,
        schedule_id: str,
        last_run_time: str,
        anomaly_count: int,
        alert_triggered: bool,
    ) -> None:
        await self._conn.execute(
            "UPDATE schedules SET last_run_time = ?, last_anomaly_count = ?, "
            "last_alert_triggered = ?, status = 'running' WHERE id = ?",
            (last_run_time, anomaly_count, int(alert_triggered), schedule_id),
        )
        await self._conn.commit()

    async def save_execution_history(self, history_dict: dict) -> int:
        cursor = await self._conn.execute(
            """INSERT INTO execution_history
               (schedule_id, start_time, status)
               VALUES (?, ?, ?)""",
            (
                history_dict["schedule_id"],
                history_dict["start_time"],
                history_dict.get("status", "running"),
            ),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def update_execution_history(self, history_id: int, update_dict: dict) -> None:
        sets: list[str] = []
        params: list = []
        for key in ("end_time", "anomaly_count", "alert_triggered", "status", "error"):
            if key in update_dict:
                sets.append(f"{key} = ?")
                val = update_dict[key]
                if key == "alert_triggered":
                    val = int(val)
                params.append(val)
        if not sets:
            return
        params.append(history_id)
        await self._conn.execute(
            f"UPDATE execution_history SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self._conn.commit()

    async def get_execution_history(
        self, schedule_id: str, limit: int = 20
    ) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT id, schedule_id, start_time, end_time, anomaly_count, "
            "alert_triggered, status, error "
            "FROM execution_history WHERE schedule_id = ? "
            "ORDER BY start_time DESC LIMIT ?",
            (schedule_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "schedule_id": r[1], "start_time": r[2],
                "end_time": r[3], "anomaly_count": r[4],
                "alert_triggered": bool(r[5]), "status": r[6], "error": r[7],
            }
            for r in rows
        ]

    async def save_model_version(self, model_dict: dict) -> None:
        await self._conn.execute(
            """INSERT INTO model_versions
               (id, name, algorithm_type, version, training_params, training_data_start,
                training_data_end, precision, recall, f1, status, parent_version_id,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_dict["id"],
                model_dict["name"],
                model_dict["algorithm_type"],
                model_dict["version"],
                model_dict.get("training_params", "{}"),
                model_dict.get("training_data_start", ""),
                model_dict.get("training_data_end", ""),
                model_dict.get("precision", 0.0),
                model_dict.get("recall", 0.0),
                model_dict.get("f1", 0.0),
                model_dict.get("status", "training"),
                model_dict.get("parent_version_id"),
                model_dict.get("created_at", datetime.utcnow().isoformat()),
                model_dict.get("updated_at", datetime.utcnow().isoformat()),
            ),
        )
        await self._conn.commit()

    async def get_model_version(self, model_id: str) -> Optional[dict]:
        cursor = await self._conn.execute(
            "SELECT id, name, algorithm_type, version, training_params, training_data_start, "
            "training_data_end, precision, recall, f1, status, parent_version_id, "
            "created_at, updated_at FROM model_versions WHERE id = ?",
            (model_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "name": row[1], "algorithm_type": row[2],
            "version": row[3], "training_params": row[4],
            "training_data_start": row[5], "training_data_end": row[6],
            "precision": row[7], "recall": row[8], "f1": row[9],
            "status": row[10], "parent_version_id": row[11],
            "created_at": row[12], "updated_at": row[13],
        }

    async def list_model_versions(self, name: str) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT id, name, algorithm_type, version, training_params, training_data_start, "
            "training_data_end, precision, recall, f1, status, parent_version_id, "
            "created_at, updated_at FROM model_versions WHERE name = ? ORDER BY created_at DESC",
            (name,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "name": r[1], "algorithm_type": r[2],
                "version": r[3], "training_params": r[4],
                "training_data_start": r[5], "training_data_end": r[6],
                "precision": r[7], "recall": r[8], "f1": r[9],
                "status": r[10], "parent_version_id": r[11],
                "created_at": r[12], "updated_at": r[13],
            }
            for r in rows
        ]

    async def list_model_groups(self) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT name, COUNT(*) as version_count, "
            "MAX(created_at) as latest_created, "
            "GROUP_CONCAT(CASE WHEN status = 'active' THEN id END) as active_model_id "
            "FROM model_versions GROUP BY name ORDER BY latest_created DESC"
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            active_id = r[3]
            if active_id and "," in active_id:
                active_id = active_id.split(",")[0]
            algo_cursor = await self._conn.execute(
                "SELECT algorithm_type, f1, version FROM model_versions WHERE name = ? AND status = 'active' LIMIT 1",
                (r[0],),
            )
            algo_row = await algo_cursor.fetchone()
            result.append({
                "name": r[0],
                "version_count": r[1],
                "latest_created": r[2],
                "active_model_id": active_id,
                "algorithm_type": algo_row[0] if algo_row else "",
                "active_f1": algo_row[1] if algo_row else 0.0,
                "active_version": algo_row[2] if algo_row else "",
            })
        return result

    async def get_active_model_version(self, name: str) -> Optional[dict]:
        cursor = await self._conn.execute(
            "SELECT id, name, algorithm_type, version, training_params, training_data_start, "
            "training_data_end, precision, recall, f1, status, parent_version_id, "
            "created_at, updated_at FROM model_versions WHERE name = ? AND status = 'active' LIMIT 1",
            (name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "name": row[1], "algorithm_type": row[2],
            "version": row[3], "training_params": row[4],
            "training_data_start": row[5], "training_data_end": row[6],
            "precision": row[7], "recall": row[8], "f1": row[9],
            "status": row[10], "parent_version_id": row[11],
            "created_at": row[12], "updated_at": row[13],
        }

    async def update_model_version_status(self, model_id: str, status: str) -> None:
        now = datetime.utcnow().isoformat()
        await self._conn.execute(
            "UPDATE model_versions SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, model_id),
        )
        await self._conn.commit()

    async def update_model_version_metrics(
        self, model_id: str, precision: float, recall: float, f1: float, updated_at: str
    ) -> None:
        await self._conn.execute(
            "UPDATE model_versions SET precision = ?, recall = ?, f1 = ?, updated_at = ? WHERE id = ?",
            (precision, recall, f1, updated_at, model_id),
        )
        await self._conn.commit()

    async def update_model_parent(self, model_id: str, parent_id: str) -> None:
        await self._conn.execute(
            "UPDATE model_versions SET parent_version_id = ? WHERE id = ?",
            (parent_id, model_id),
        )
        await self._conn.commit()

    async def delete_model_version(self, model_id: str) -> None:
        model_row = await self.get_model_version(model_id)
        model_name = model_row["name"] if model_row else None
        if model_name:
            await self._conn.execute(
                "DELETE FROM retrain_configs WHERE model_name = ?",
                (model_name,),
            )
        await self._conn.execute(
            "DELETE FROM model_f1_history WHERE model_id = ?",
            (model_id,),
        )
        await self._conn.execute(
            "DELETE FROM training_contexts WHERE model_id = ?",
            (model_id,),
        )
        await self._conn.execute(
            "DELETE FROM model_versions WHERE id = ?",
            (model_id,),
        )
        await self._conn.commit()

    async def save_retrain_config(self, config_dict: dict) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO retrain_configs
               (model_name, trigger_type, scheduled_interval_hours, performance_window_size,
                performance_f1_threshold, drift_kl_threshold, training_data_days, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config_dict["model_name"],
                config_dict.get("trigger_type", "scheduled"),
                config_dict.get("scheduled_interval_hours", 24),
                config_dict.get("performance_window_size", 10),
                config_dict.get("performance_f1_threshold", 0.7),
                config_dict.get("drift_kl_threshold", 0.5),
                config_dict.get("training_data_days", 30),
                int(config_dict.get("enabled", True)),
            ),
        )
        await self._conn.commit()

    async def get_retrain_config(self, model_name: str) -> Optional[dict]:
        cursor = await self._conn.execute(
            "SELECT model_name, trigger_type, scheduled_interval_hours, performance_window_size, "
            "performance_f1_threshold, drift_kl_threshold, training_data_days, enabled "
            "FROM retrain_configs WHERE model_name = ?",
            (model_name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "model_name": row[0], "trigger_type": row[1],
            "scheduled_interval_hours": row[2],
            "performance_window_size": row[3],
            "performance_f1_threshold": row[4],
            "drift_kl_threshold": row[5],
            "training_data_days": row[6],
            "enabled": bool(row[7]),
        }

    async def list_retrain_configs(self) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT model_name, trigger_type, scheduled_interval_hours, performance_window_size, "
            "performance_f1_threshold, drift_kl_threshold, training_data_days, enabled "
            "FROM retrain_configs"
        )
        rows = await cursor.fetchall()
        return [
            {
                "model_name": r[0], "trigger_type": r[1],
                "scheduled_interval_hours": r[2],
                "performance_window_size": r[3],
                "performance_f1_threshold": r[4],
                "drift_kl_threshold": r[5],
                "training_data_days": r[6],
                "enabled": bool(r[7]),
            }
            for r in rows
        ]

    async def save_training_context(self, ctx_dict: dict) -> int:
        import json as _json
        cursor = await self._conn.execute(
            """INSERT INTO training_contexts
               (model_id, training_data_count, training_duration_seconds, stages,
                old_precision, old_recall, old_f1, new_precision, new_recall, new_f1,
                auto_activated, error, completed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ctx_dict["model_id"],
                ctx_dict.get("training_data_count", 0),
                ctx_dict.get("training_duration_seconds", 0.0),
                _json.dumps(ctx_dict.get("stages", [])),
                ctx_dict.get("old_precision", 0.0),
                ctx_dict.get("old_recall", 0.0),
                ctx_dict.get("old_f1", 0.0),
                ctx_dict.get("new_precision", 0.0),
                ctx_dict.get("new_recall", 0.0),
                ctx_dict.get("new_f1", 0.0),
                int(ctx_dict.get("auto_activated", False)),
                ctx_dict.get("error"),
                ctx_dict.get("completed_at", ""),
                datetime.utcnow().isoformat(),
            ),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def get_training_contexts(self, model_id: str, limit: int = 20) -> list[dict]:
        import json as _json
        cursor = await self._conn.execute(
            "SELECT id, model_id, training_data_count, training_duration_seconds, stages, "
            "old_precision, old_recall, old_f1, new_precision, new_recall, new_f1, "
            "auto_activated, error, completed_at, created_at "
            "FROM training_contexts WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
            (model_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "model_id": r[1],
                "training_data_count": r[2],
                "training_duration_seconds": r[3],
                "stages": _json.loads(r[4]) if r[4] else [],
                "old_precision": r[5], "old_recall": r[6], "old_f1": r[7],
                "new_precision": r[8], "new_recall": r[9], "new_f1": r[10],
                "auto_activated": bool(r[11]),
                "error": r[12], "completed_at": r[13], "created_at": r[14],
            }
            for r in rows
        ]

    async def save_f1_score(self, model_id: str, f1_score: float) -> None:
        now = datetime.utcnow().isoformat()
        await self._conn.execute(
            "INSERT INTO model_f1_history (model_id, f1_score, recorded_at) VALUES (?, ?, ?)",
            (model_id, f1_score, now),
        )
        await self._conn.commit()

    async def get_recent_f1_scores(self, model_id: str, limit: int = 10) -> list[float]:
        cursor = await self._conn.execute(
            "SELECT f1_score FROM model_f1_history WHERE model_id = ? ORDER BY recorded_at DESC LIMIT ?",
            (model_id, limit),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def get_last_training_time(self, model_id: str) -> Optional[str]:
        cursor = await self._conn.execute(
            "SELECT completed_at FROM training_contexts WHERE model_id = ? AND error IS NULL ORDER BY completed_at DESC LIMIT 1",
            (model_id,),
        )
        row = await cursor.fetchone()
        if row is None or not row[0]:
            return None
        return row[0]
