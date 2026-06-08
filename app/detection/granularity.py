import numpy as np
import pandas as pd
import ruptures as rpt

from app.detection.base import AnomalyResult, AnomalyType, BaseDetector


class PointAnomalyDetector(BaseDetector):
    def __init__(self, detector: BaseDetector):
        self._detector = detector

    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        results = self._detector.detect(series, config)
        for r in results:
            r.anomaly_type = AnomalyType.POINT
        return results


class ContextualAnomalyDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        window_size: int = config.get("context_window", 50)
        z_threshold: float = config.get("context_z_threshold", 2.5)
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        if isinstance(timestamps, pd.DatetimeIndex):
            detrended = self._detrend_by_time_context(values, timestamps, config)
        else:
            detrended = self._detrend_rolling(values, window_size)

        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            window = detrended[start : i + 1]
            w_mean = np.mean(window)
            w_std = np.std(window)
            if w_std == 0:
                is_anomaly = False
                score = 0.0
            else:
                score = abs(detrended[i] - w_mean) / w_std
                is_anomaly = score > z_threshold
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=score,
                    algorithm_name="contextual",
                    anomaly_type=AnomalyType.CONTEXTUAL,
                )
            )
        return results

    @staticmethod
    def _detrend_rolling(values: np.ndarray, window_size: int) -> np.ndarray:
        s = pd.Series(values)
        rolling_mean = s.rolling(window=window_size, min_periods=1, center=True).mean()
        return (values - rolling_mean.values).astype(float)

    @staticmethod
    def _detrend_by_time_context(
        values: np.ndarray,
        timestamps: pd.DatetimeIndex,
        config: dict,
    ) -> np.ndarray:
        group_by: str = config.get("context_group_by", "hour")
        df = pd.DataFrame({"value": values}, index=timestamps)
        if group_by == "hour":
            df["group"] = df.index.hour
        elif group_by == "dow_hour":
            df["group"] = df.index.dayofweek.astype(str) + "_" + df.index.hour.astype(str)
        else:
            df["group"] = df.index.hour
        group_means = df.groupby("group")["value"].transform("mean")
        return (values - group_means.values).astype(float)


class CollectiveAnomalyDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        slope_window: int = config.get("slope_window", 10)
        slope_threshold: float = config.get("slope_threshold", 2.0)
        pelt_penalty: float = config.get("pelt_penalty", 10.0)
        min_segment_length: int = config.get("min_segment_length", 5)
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index
        n = len(values)

        slope_scores = self._compute_slope_scores(values, slope_window)
        changepoints = self._detect_changepoints(values, pelt_penalty, min_segment_length)
        anomaly_segments = self._identify_anomaly_segments(
            values, slope_scores, changepoints, slope_threshold
        )

        is_anomaly_flags = np.zeros(n, dtype=bool)
        score_arr = np.zeros(n, dtype=float)
        for seg_start, seg_end in anomaly_segments:
            is_anomaly_flags[seg_start:seg_end] = True
            for i in range(seg_start, seg_end):
                score_arr[i] = slope_scores[i] if slope_scores[i] > 0 else 1.0

        for i in range(n):
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=bool(is_anomaly_flags[i]),
                    score=float(score_arr[i]),
                    algorithm_name="collective",
                    anomaly_type=AnomalyType.COLLECTIVE,
                )
            )
        return results

    @staticmethod
    def _compute_slope_scores(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        scores = np.zeros(n, dtype=float)
        if n < window:
            return scores
        for i in range(window - 1, n):
            segment = values[i - window + 1 : i + 1]
            x = np.arange(window, dtype=float)
            mean_x = np.mean(x)
            mean_y = np.mean(segment)
            numerator = np.sum((x - mean_x) * (segment - mean_y))
            denominator = np.sum((x - mean_x) ** 2)
            if denominator == 0:
                slope = 0.0
            else:
                slope = numerator / denominator
            std_y = np.std(segment)
            scores[i] = abs(slope) / std_y if std_y > 0 else 0.0
        for i in range(window - 1):
            scores[i] = scores[window - 1] if window - 1 < n else 0.0
        return scores

    @staticmethod
    def _detect_changepoints(
        values: np.ndarray, penalty: float, min_segment: int
    ) -> list[int]:
        if len(values) < 2 * min_segment:
            return []
        signal = values.reshape(-1, 1).astype(float)
        algo = rpt.Pelt(model="rbf", min_size=min_segment).fit(signal)
        try:
            cps = algo.predict(pen=penalty)
        except Exception:
            return []
        return [cp for cp in cps if cp < len(values)]

    @staticmethod
    def _identify_anomaly_segments(
        values: np.ndarray,
        slope_scores: np.ndarray,
        changepoints: list[int],
        slope_threshold: float,
    ) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        boundaries = [0] + changepoints + [len(values)]
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_slopes = slope_scores[seg_start:seg_end]
            if len(seg_slopes) == 0:
                continue
            mean_slope_score = np.mean(seg_slopes)
            if mean_slope_score > slope_threshold:
                segments.append((seg_start, seg_end))
        return segments
