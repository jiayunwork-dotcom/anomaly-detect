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
