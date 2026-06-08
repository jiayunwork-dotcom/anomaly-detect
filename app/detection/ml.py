import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.ensemble import IsolationForest

from app.detection.base import AnomalyResult, AnomalyType, BaseDetector


class IsolationForestDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        contamination: float = config.get("contamination", 0.05)
        n_estimators: int = config.get("n_estimators", 100)
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        X = values.reshape(-1, 1)
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        predictions = clf.fit_predict(X)
        scores = -clf.score_samples(X)

        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        normalized_scores = scores / max_score

        for i in range(len(values)):
            is_anomaly = predictions[i] == -1
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=float(normalized_scores[i]),
                    algorithm_name="isolation_forest",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results


class _LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        seq_len = x.size(1)
        decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder(decoder_input)
        return output


class LSTMEncoderDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        window_size: int = config.get("window", 30)
        hidden_dim: int = config.get("hidden_dim", 32)
        num_layers: int = config.get("num_layers", 1)
        epochs: int = config.get("epochs", 20)
        learning_rate: float = config.get("learning_rate", 1e-3)
        threshold_percentile: int = config.get("threshold_percentile", 99)
        threshold_window: int = config.get("threshold_window", 50)

        values = series.values.astype(float)
        timestamps = series.index
        n = len(values)

        if n < window_size:
            return [
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=False,
                    score=0.0,
                    algorithm_name="lstm_autoencoder",
                    anomaly_type=AnomalyType.POINT,
                )
                for i in range(n)
            ]

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1.0
        normalized = (values - mean) / std

        windows: list[np.ndarray] = []
        for i in range(n - window_size + 1):
            windows.append(normalized[i : i + window_size])

        windows_arr = np.array(windows, dtype=np.float32)
        X = torch.from_numpy(windows_arr).unsqueeze(-1)

        model = _LSTMAutoencoder(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            reconstructed = model(X).numpy()

        reconstruction_errors = np.mean((windows_arr - reconstructed.squeeze(-1)) ** 2, axis=1)

        point_errors = np.zeros(n, dtype=float)
        point_counts = np.zeros(n, dtype=float)
        for i in range(len(reconstruction_errors)):
            for j in range(window_size):
                idx = i + j
                point_errors[idx] += reconstruction_errors[i]
                point_counts[idx] += 1
        point_errors = point_errors / np.maximum(point_counts, 1.0)

        adaptive_thresholds = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - threshold_window)
            window_errors = point_errors[start : i + 1]
            adaptive_thresholds[i] = np.percentile(window_errors, threshold_percentile)

        results: list[AnomalyResult] = []
        for i in range(n):
            is_anomaly = point_errors[i] > adaptive_thresholds[i] and adaptive_thresholds[i] > 0
            score = point_errors[i] / adaptive_thresholds[i] if adaptive_thresholds[i] > 0 else 0.0
            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=float(score),
                    algorithm_name="lstm_autoencoder",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results


class ProphetDetector(BaseDetector):
    def detect(self, series: pd.Series, config: dict) -> list[AnomalyResult]:
        interval_width: float = config.get("interval_width", 0.95)
        results: list[AnomalyResult] = []
        values = series.values.astype(float)
        timestamps = series.index

        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        df = pd.DataFrame({"ds": timestamps, "y": values})

        model = Prophet(interval_width=interval_width)
        model.fit(df)

        forecast = model.predict(df)

        for i in range(len(values)):
            yhat_lower = forecast.iloc[i]["yhat_lower"]
            yhat_upper = forecast.iloc[i]["yhat_upper"]
            yhat = forecast.iloc[i]["yhat"]
            actual = values[i]

            range_width = yhat_upper - yhat_lower
            if range_width == 0:
                is_anomaly = False
                score = 0.0
            else:
                distance = max(yhat_lower - actual, actual - yhat_upper, 0)
                score = distance / range_width
                is_anomaly = actual < yhat_lower or actual > yhat_upper

            results.append(
                AnomalyResult(
                    timestamp=timestamps[i],
                    is_anomaly=is_anomaly,
                    score=score,
                    algorithm_name="prophet",
                    anomaly_type=AnomalyType.POINT,
                )
            )
        return results
