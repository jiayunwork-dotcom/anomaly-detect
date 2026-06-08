import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from app.detection.base import AnomalyResult, AnomalyType, BaseDetector


class ThreeSigmaDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        window_size: int = config.get("window_size", 100)
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            window = values[start : i + 1]
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                is_anomaly = False
                score = 0.0
            else:
                score = abs(values[i] - mean) / std
                is_anomaly = values[i] > mean + 3 * std or values[i] < mean - 3 * std
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=score,
                    algorithm_name="three_sigma",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results


class IQRFecutor(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_range = upper - lower

        for i in range(len(values)):
            if iqr_range == 0:
                is_anomaly = False
                score = 0.0
            else:
                distance = max(lower - values[i], values[i] - upper, 0)
                score = distance / iqr_range
                is_anomaly = values[i] < lower or values[i] > upper
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=score,
                    algorithm_name="iqr",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results


class STLDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        period: int | None = config.get("period")
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        if period is None:
            period = self._infer_period(timestamps)

        if period is None or period < 2:
            for i in range(len(values)):
                results.append(
                    AnomalyResult(
                        timestamp=timestamps[i],
                        is_anomaly=False,
                        score=0.0,
                        algorithm_name="stl",
                        anomaly_type=AnomalyType.POINT,
                    )
                )
            return results

        stl_series = pd.Series(values, index=pd.RangeIndex(len(values)))
        stl = STL(stl_series, period=period, robust=True)
        res = stl.fit()
        residuals = res.resid

        resid_mean = np.mean(residuals)
        resid_std = np.std(residuals)

        for i in range(len(values)):
            if resid_std == 0:
                is_anomaly = False
                score = 0.0
            else:
                score = abs(residuals[i] - resid_mean) / resid_std
                is_anomaly = (
                    residuals[i] > resid_mean + 3 * resid_std
                    or residuals[i] < resid_mean - 3 * resid_std
                )
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=score,
                    algorithm_name="stl",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results

    @staticmethod
    def _infer_period(timestamps: pd.Index) -> int | None:
        if not isinstance(timestamps, pd.DatetimeIndex):
            return None
        if len(timestamps) < 3:
            return None
        diffs = pd.Series(timestamps).diff().dropna()
        if diffs.empty:
            return None
        median_diff = diffs.median()
        total_span = timestamps[-1] - timestamps[0]
        if median_diff.total_seconds() == 0:
            return None
        estimated_period = int(total_span / median_diff)
        if estimated_period < 2:
            return None
        return min(estimated_period, len(timestamps) // 2)
