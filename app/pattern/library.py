from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from dtw import dtw as dtw_align
from scipy import stats as sp_stats

from app.storage.database import StorageManager


@dataclass
class PatternFingerprint:
    metrics: list[str]
    shape: str
    duration_seconds: float
    root_cause: str
    segment_values: list[float] = field(default_factory=list)
    pattern_id: Optional[int] = None


def classify_shape(series_segment: pd.Series) -> str:
    if len(series_segment) < 3:
        return "spike"

    values = series_segment.values.astype(float)
    n = len(values)
    mid = n // 2

    first_diff = np.diff(values)
    mean_first_diff = float(np.mean(first_diff))

    first_half_mean = float(np.mean(values[:mid + 1]))
    second_half_mean = float(np.mean(values[mid:]))
    peak_idx = int(np.argmax(values))
    trough_idx = int(np.argmin(values))

    peak_near_center = abs(peak_idx - mid) <= max(1, n // 4)
    trough_near_center = abs(trough_idx - mid) <= max(1, n // 4)

    is_spike = (
        peak_near_center
        and peak_idx != 0
        and peak_idx != n - 1
        and values[peak_idx] > first_half_mean
        and values[peak_idx] > second_half_mean
    )

    is_drop = (
        trough_near_center
        and trough_idx != 0
        and trough_idx != n - 1
        and values[trough_idx] < first_half_mean
        and values[trough_idx] < second_half_mean
    )

    sign_changes = int(np.sum(np.diff(np.sign(first_diff)) != 0))
    oscillation_ratio = sign_changes / max(len(first_diff), 1)

    is_oscillation = oscillation_ratio >= 0.5

    if is_oscillation:
        return "oscillation"

    if is_spike and is_drop:
        if values[peak_idx] - np.min(values) > np.max(values) - values[trough_idx]:
            return "spike"
        else:
            return "drop"

    if is_spike:
        return "spike"

    if is_drop:
        return "drop"

    slope, _, _, _, _ = sp_stats.linregress(np.arange(n), values)
    if slope > 0 and mean_first_diff > 0:
        return "gradual_rise"

    if slope < 0 and mean_first_diff < 0:
        return "drop"

    return "spike"


def _compute_dtw_distance(
    series_a: np.ndarray, series_b: np.ndarray
) -> float:
    a = series_a.astype(np.float64).reshape(-1, 1)
    b = series_b.astype(np.float64).reshape(-1, 1)

    range_a = float(np.ptp(a)) if np.ptp(a) > 0 else 1.0
    range_b = float(np.ptp(b)) if np.ptp(b) > 0 else 1.0
    scale = max(range_a, range_b)

    a_norm = a / scale
    b_norm = b / scale

    alignment = dtw_align(a_norm, b_norm)
    raw_distance = float(alignment.distance)
    normalized = raw_distance / max(len(a), len(b))
    return float(min(normalized, 1.0))


def _compute_jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class PatternLibrary:
    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    async def extract_fingerprint(
        self,
        metrics: list[str],
        anomaly_segment: pd.Series,
        duration_seconds: float,
        root_cause: str,
    ) -> PatternFingerprint:
        shape = classify_shape(anomaly_segment)
        segment_values = anomaly_segment.values.astype(float).tolist()

        return PatternFingerprint(
            metrics=metrics,
            shape=shape,
            duration_seconds=duration_seconds,
            root_cause=root_cause,
            segment_values=segment_values,
        )

    async def save_fingerprint(
        self,
        fingerprint: PatternFingerprint,
        name: str,
        resolution: str = "",
    ) -> int:
        pattern_dict = {
            "name": name,
            "metrics_json": json.dumps(fingerprint.metrics),
            "shape": fingerprint.shape,
            "duration_seconds": fingerprint.duration_seconds,
            "root_cause": fingerprint.root_cause,
            "resolution": resolution,
        }
        pattern_id = await self.storage.save_pattern(pattern_dict)
        fingerprint.pattern_id = pattern_id
        return pattern_id

    async def match_similarity(
        self,
        metrics: list[str],
        anomaly_segment: pd.Series,
        similarity_threshold: float = 0.7,
    ) -> list[dict]:
        query_segment = anomaly_segment.values.astype(np.float64).ravel()
        query_metrics_set = set(metrics)

        cursor = await self.storage._conn.execute(
            "SELECT id, name, metrics_json, shape, duration_seconds, root_cause, resolution, created_at "
            "FROM pattern_library"
        )
        rows = await cursor.fetchall()

        results: list[dict] = []

        for r in rows:
            pattern_id = r[0]
            pattern_name = r[1]
            pattern_metrics = set(json.loads(r[2]))
            pattern_shape = r[3]
            pattern_duration = r[4]
            pattern_root_cause = r[5]
            pattern_resolution = r[6]
            pattern_created = r[7]

            jaccard = _compute_jaccard(query_metrics_set, pattern_metrics)

            candidate_segment = _reconstruct_segment(
                pattern_shape, pattern_duration
            )
            if candidate_segment is not None and len(candidate_segment) > 1:
                dtw_dist = _compute_dtw_distance(query_segment, candidate_segment)
            else:
                dtw_dist = 0.0

            combined = 0.6 * jaccard + 0.4 * (1.0 - dtw_dist)

            if combined >= similarity_threshold:
                results.append({
                    "pattern_id": pattern_id,
                    "name": pattern_name,
                    "metrics": sorted(pattern_metrics),
                    "shape": pattern_shape,
                    "duration_seconds": pattern_duration,
                    "root_cause": pattern_root_cause,
                    "resolution": pattern_resolution,
                    "created_at": pattern_created,
                    "similarity": combined,
                    "metric_jaccard": jaccard,
                    "dtw_distance": dtw_dist,
                    "tag": f"suspected recurrence of historical event {pattern_name}",
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    async def confirm_and_store(
        self,
        metrics: list[str],
        anomaly_segment: pd.Series,
        duration_seconds: float,
        root_cause: str,
        name: str,
        resolution: str = "",
    ) -> PatternFingerprint:
        fingerprint = await self.extract_fingerprint(
            metrics=metrics,
            anomaly_segment=anomaly_segment,
            duration_seconds=duration_seconds,
            root_cause=root_cause,
        )
        pattern_id = await self.save_fingerprint(
            fingerprint=fingerprint,
            name=name,
            resolution=resolution,
        )
        fingerprint.pattern_id = pattern_id
        return fingerprint


def _reconstruct_segment(
    shape: str, duration_seconds: float
) -> Optional[np.ndarray]:
    n = max(int(duration_seconds), 10)
    t = np.linspace(0.0, 1.0, n)

    if shape == "spike":
        peak = np.exp(-((t - 0.5) ** 2) / 0.02)
        return peak
    elif shape == "drop":
        drop = -np.exp(-((t - 0.5) ** 2) / 0.02)
        return drop
    elif shape == "gradual_rise":
        return t
    elif shape == "oscillation":
        return np.sin(2.0 * np.pi * 3.0 * t)
    else:
        return None
