from collections import defaultdict

import pandas as pd

from app.detection.base import AnomalyResult, AnomalyType, BaseDetector


class EnsembleDetector(BaseDetector):
    def __init__(
        self,
        detectors: list[BaseDetector],
        weights: dict[str, float] | None = None,
    ):
        self._detectors = detectors
        self._weights = weights or {}

    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        mode: str = config.get("ensemble_mode", "majority")
        all_results: list[list[AnomalyResult]] = []

        for detector in self._detectors:
            detector_results = detector.detect(series, config)
            all_results.append(detector_results)

        if not all_results:
            return []

        n = len(all_results[0])

        if mode == "majority":
            return self._majority_vote(all_results, n)
        elif mode == "weighted":
            return self._weighted_vote(all_results, n)
        else:
            return self._majority_vote(all_results, n)

    @staticmethod
    def _majority_vote(
        all_results: list[list[AnomalyResult]], n: int
    ) -> list[AnomalyResult]:
        results: list[AnomalyResult] = []
        num_detectors = len(all_results)

        for i in range(n):
            anomaly_count = 0
            total_score = 0.0
            anomaly_types: list[AnomalyType] = []
            timestamp = all_results[0][i].timestamp

            for detector_results in all_results:
                r = detector_results[i]
                if r.is_anomaly:
                    anomaly_count += 1
                total_score += r.score
                anomaly_types.append(r.anomaly_type)

            is_anomaly = anomaly_count > num_detectors / 2
            avg_score = total_score / num_detectors
            most_common_type = _most_common_anomaly_type(anomaly_types)

            results.append(
                AnomalyResult(
                    timestamp=timestamp,
                    is_anomaly=is_anomaly,
                    score=avg_score,
                    algorithm_name="ensemble_majority",
                    anomaly_type=most_common_type,
                )
            )
        return results

    def _weighted_vote(
        self, all_results: list[list[AnomalyResult]], n: int
    ) -> list[AnomalyResult]:
        results: list[AnomalyResult] = []

        for i in range(n):
            weighted_anomaly = 0.0
            weighted_score = 0.0
            total_weight = 0.0
            anomaly_types: list[AnomalyType] = []
            timestamp = all_results[0][i].timestamp

            for detector_results in all_results:
                r = detector_results[i]
                weight = self._weights.get(r.algorithm_name, 1.0)
                if r.is_anomaly:
                    weighted_anomaly += weight
                weighted_score += r.score * weight
                total_weight += weight
                anomaly_types.append(r.anomaly_type)

            if total_weight == 0:
                is_anomaly = False
                avg_score = 0.0
            else:
                is_anomaly = weighted_anomaly / total_weight > 0.5
                avg_score = weighted_score / total_weight

            most_common_type = _most_common_anomaly_type(anomaly_types)

            results.append(
                AnomalyResult(
                    timestamp=timestamp,
                    is_anomaly=is_anomaly,
                    score=avg_score,
                    algorithm_name="ensemble_weighted",
                    anomaly_type=most_common_type,
                )
            )
        return results


def merge_granularity_results(
    point_results: list[AnomalyResult],
    contextual_results: list[AnomalyResult],
    collective_results: list[AnomalyResult],
) -> list[AnomalyResult]:
    if not point_results and not contextual_results and not collective_results:
        return []

    all_timestamps: set = set()
    for results in (point_results, contextual_results, collective_results):
        for r in results:
            all_timestamps.add(r.timestamp)

    ts_to_results: dict = defaultdict(list)
    for results in (point_results, contextual_results, collective_results):
        for r in results:
            ts_to_results[r.timestamp].append(r)

    merged: list[AnomalyResult] = []
    for ts in sorted(ts_to_results.keys()):
        entries = ts_to_results[ts]
        is_anomaly = any(e.is_anomaly for e in entries)
        max_score = max(e.score for e in entries)
        types = [e.anomaly_type for e in entries if e.is_anomaly]
        if types:
            anomaly_type = _most_common_anomaly_type(types)
        else:
            anomaly_type = AnomalyType.POINT

        merged.append(
            AnomalyResult(
                timestamp=ts,
                is_anomaly=is_anomaly,
                score=max_score,
                algorithm_name="ensemble_merged",
                anomaly_type=anomaly_type,
            )
        )
    return merged


def _most_common_anomaly_type(types: list[AnomalyType]) -> AnomalyType:
    if not types:
        return AnomalyType.POINT
    counts: dict[AnomalyType, int] = defaultdict(int)
    for t in types:
        counts[t] += 1
    return max(counts, key=counts.get)
