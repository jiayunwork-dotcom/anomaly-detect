from __future__ import annotations

from datetime import datetime
from typing import Any

from app.storage.database import StorageManager


class LabelingLoop:
    def __init__(
        self,
        storage: StorageManager,
        base_weights: dict[str, float] | None = None,
        threshold_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._storage = storage
        self._base_weights: dict[str, float] = base_weights or {}
        self._thresholds: dict[str, dict[str, float]] = {}
        self._init_default_thresholds()
        if threshold_overrides:
            for algo, params in threshold_overrides.items():
                self._thresholds[algo] = dict(params)

    def _init_default_thresholds(self) -> None:
        self._thresholds = {
            "three_sigma": {"sigma_multiplier": 3.0},
            "iqr": {"iqr_multiplier": 1.5},
            "stl": {"sigma_multiplier": 3.0},
            "isolation_forest": {"contamination": 0.05},
            "lstm_autoencoder": {"threshold_percentile": 99.0},
            "prophet": {"interval_width": 0.95},
        }

    async def _get_event_by_id(self, event_id: int) -> dict[str, Any] | None:
        cursor = await self._storage._conn.execute(
            "SELECT id, metric_name, start_time, end_time, anomaly_type, severity, "
            "algorithm, is_confirmed, confirmed_as, pattern_id, created_at "
            "FROM anomaly_events WHERE id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "metric_name": row[1],
            "start_time": row[2],
            "end_time": row[3],
            "anomaly_type": row[4],
            "severity": row[5],
            "algorithm": row[6],
            "is_confirmed": bool(row[7]),
            "confirmed_as": row[8],
            "pattern_id": row[9],
            "created_at": row[10],
        }

    async def _estimate_fn_counts(self) -> dict[str, int]:
        cursor = await self._storage._conn.execute(
            "SELECT metric_name, start_time, end_time, algorithm "
            "FROM anomaly_events WHERE is_confirmed = 1 AND confirmed_as = 'TP'"
        )
        confirmed_tp_rows = await cursor.fetchall()

        cursor = await self._storage._conn.execute(
            "SELECT DISTINCT algorithm FROM anomaly_events WHERE algorithm != ''"
        )
        algo_rows = await cursor.fetchall()
        algorithms = [r[0] for r in algo_rows]

        cursor = await self._storage._conn.execute(
            "SELECT metric_name, start_time, end_time, algorithm FROM anomaly_events"
        )
        all_events = await cursor.fetchall()

        fn_counts: dict[str, int] = {algo: 0 for algo in algorithms}

        for tp_metric, tp_start_str, tp_end_str, tp_algo in confirmed_tp_rows:
            tp_start = datetime.fromisoformat(tp_start_str)
            tp_end = datetime.fromisoformat(tp_end_str)

            for algo in algorithms:
                if algo == tp_algo:
                    continue
                has_overlap = False
                for ev_metric, ev_start_str, ev_end_str, ev_algo in all_events:
                    if ev_algo != algo or ev_metric != tp_metric:
                        continue
                    ev_start = datetime.fromisoformat(ev_start_str)
                    ev_end = datetime.fromisoformat(ev_end_str)
                    if ev_start <= tp_end and ev_end >= tp_start:
                        has_overlap = True
                        break
                if not has_overlap:
                    fn_counts[algo] += 1

        return fn_counts

    async def confirm_event(self, event_id: int) -> None:
        event = await self._get_event_by_id(event_id)
        if event is None:
            return
        await self._storage.update_anomaly_confirmation(event_id, "TP")
        algorithm = event["algorithm"]
        if algorithm:
            await self._storage.update_algorithm_performance(algorithm, tp=1, fp=0)

    async def mark_false_alarm(self, event_id: int) -> None:
        event = await self._get_event_by_id(event_id)
        if event is None:
            return
        await self._storage.update_anomaly_confirmation(event_id, "FP")
        algorithm = event["algorithm"]
        if algorithm:
            await self._storage.update_algorithm_performance(algorithm, tp=0, fp=1)

    async def adjust_thresholds(self) -> dict[str, dict[str, float]]:
        performances = await self._storage.get_algorithm_performance()
        adjustments: dict[str, dict[str, float]] = {}

        for perf in performances:
            algo = perf["algorithm_name"]
            tp = perf["tp_count"]
            fp = perf["fp_count"]
            total = tp + fp
            if total == 0:
                continue
            fp_ratio = fp / total

            if algo not in self._thresholds:
                continue

            if fp_ratio > 0.3:
                self._tighten_threshold(algo)
            elif fp_ratio < 0.1:
                self._relax_threshold(algo)

            adjustments[algo] = dict(self._thresholds[algo])

        return adjustments

    def _tighten_threshold(self, algo: str) -> None:
        current = self._thresholds[algo]
        if "sigma_multiplier" in current:
            current["sigma_multiplier"] *= 1.1
        if "iqr_multiplier" in current:
            current["iqr_multiplier"] *= 1.1
        if "contamination" in current:
            current["contamination"] *= 0.9
        if "threshold_percentile" in current:
            current["threshold_percentile"] = min(
                current["threshold_percentile"] * 1.01, 99.9
            )
        if "interval_width" in current:
            current["interval_width"] = min(current["interval_width"] * 1.01, 0.99)

    def _relax_threshold(self, algo: str) -> None:
        current = self._thresholds[algo]
        if "sigma_multiplier" in current:
            current["sigma_multiplier"] *= 0.95
        if "iqr_multiplier" in current:
            current["iqr_multiplier"] *= 0.95
        if "contamination" in current:
            current["contamination"] *= 1.05
        if "threshold_percentile" in current:
            current["threshold_percentile"] *= 0.99
        if "interval_width" in current:
            current["interval_width"] *= 0.99

    async def compute_voting_weights(self) -> dict[str, float]:
        performances = await self._storage.get_algorithm_performance()
        weights: dict[str, float] = {}

        for perf in performances:
            algo = perf["algorithm_name"]
            tp = perf["tp_count"]
            fp = perf["fp_count"]
            base = self._base_weights.get(algo, 1.0)
            fp_penalty = min(fp / (tp + fp + 1), 0.5)
            weights[algo] = base * (1 - fp_penalty)

        return weights

    async def compute_algorithm_performance(self) -> dict[str, dict[str, float]]:
        performances = await self._storage.get_algorithm_performance()
        fn_counts = await self._estimate_fn_counts()
        result: dict[str, dict[str, float]] = {}

        for perf in performances:
            algo = perf["algorithm_name"]
            tp = perf["tp_count"]
            fp = perf["fp_count"]
            fn = fn_counts.get(algo, 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            result[algo] = {
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "precision": precision,
                "recall": recall,
            }

        return result

    async def export_labels(
        self, start_time: datetime, end_time: datetime, output_path: str
    ) -> None:
        await self._storage.export_labels_csv(start_time, end_time, output_path)

    async def get_algorithm_stats(self) -> dict[str, dict[str, Any]]:
        performances = await self._storage.get_algorithm_performance()
        fn_counts = await self._estimate_fn_counts()
        stats: dict[str, dict[str, Any]] = {}

        for perf in performances:
            algo = perf["algorithm_name"]
            tp = perf["tp_count"]
            fp = perf["fp_count"]
            fn = fn_counts.get(algo, 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            base = self._base_weights.get(algo, 1.0)
            fp_penalty = min(fp / (tp + fp + 1), 0.5)
            recommended_weight = base * (1 - fp_penalty)

            now = datetime.utcnow().isoformat()
            await self._storage._conn.execute(
                "UPDATE algorithm_performance SET recall_val = ?, last_updated = ? "
                "WHERE algorithm_name = ?",
                (recall, now, algo),
            )

            stats[algo] = {
                "tp": tp,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "recommended_weight": recommended_weight,
            }

        await self._storage._conn.commit()
        return stats

    def get_thresholds(self) -> dict[str, dict[str, float]]:
        return {algo: dict(params) for algo, params in self._thresholds.items()}

    def set_algorithm_threshold(
        self, algorithm: str, params: dict[str, float]
    ) -> None:
        self._thresholds[algorithm] = dict(params)
