import json
from io import StringIO

import numpy as np
import pandas as pd

MAX_METRICS = 200
MAX_DATA_POINTS = 1_000_000

FREQUENCY_MAP: dict[str, pd.Timedelta] = {
    "1s": pd.Timedelta(seconds=1),
    "5s": pd.Timedelta(seconds=5),
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(hours=1),
}


def detect_frequency(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "1m"
    deltas = pd.Series(index[1:] - index[:-1])
    median_delta = deltas.median()
    best_freq = "1m"
    best_diff = abs(median_delta - FREQUENCY_MAP["1m"])
    for freq_name, freq_td in FREQUENCY_MAP.items():
        diff = abs(median_delta - freq_td)
        if diff < best_diff:
            best_diff = diff
            best_freq = freq_name
    return best_freq


def interpolate_with_holes(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df
    df = df.copy()
    df.index = df.index.round(freq)
    df = df[~df.index.duplicated(keep="last")]
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    for col in df.columns:
        is_nan = df[col].isna()
        if not is_nan.any():
            continue
        interpolated = df[col].interpolate(method="linear")
        groups = (is_nan != is_nan.shift()).cumsum()
        for group_id in groups[is_nan].unique():
            gap_size = (groups == group_id).sum()
            if gap_size > 5:
                df.loc[groups == group_id, col] = np.nan
            else:
                df.loc[groups == group_id, col] = interpolated.loc[groups == group_id]
    return df


def _build_metadata(
    columns: pd.Index,
    source: str,
    freq: str,
    valid_ranges: dict[str, tuple[float, float]] | None,
    units: dict[str, str] | None,
) -> list[dict]:
    metadata: list[dict] = []
    for col in columns:
        meta: dict = {
            "name": col,
            "unit": (units or {}).get(col, ""),
            "source": source,
            "frequency": freq,
            "valid_range": (valid_ranges or {}).get(col, (np.nan, np.nan)),
            "out_of_range": [],
        }
        metadata.append(meta)
    return metadata


def _flag_out_of_range(df: pd.DataFrame, metadata: list[dict]) -> None:
    for meta in metadata:
        name = meta["name"]
        if name not in df.columns:
            continue
        lo, hi = meta["valid_range"]
        if np.isnan(lo) and np.isnan(hi):
            continue
        series = df[name]
        mask = pd.Series(False, index=series.index)
        if not np.isnan(lo):
            mask |= series < lo
        if not np.isnan(hi):
            mask |= series > hi
        meta["out_of_range"] = series.index[mask].tolist()


def parse_csv(
    content: str,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
    units: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(StringIO(content))
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format="mixed", utc=True)
    df = df.set_index(df.columns[0])
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    freq = detect_frequency(df.index)
    df = interpolate_with_holes(df, freq)
    metadata = _build_metadata(df.columns, "csv", freq, valid_ranges, units)
    _flag_out_of_range(df, metadata)
    return df, metadata


def parse_prometheus_json(
    content: str,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
    units: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    data = json.loads(content)
    results = data["data"]["result"]
    series_dict: dict[str, dict[pd.Timestamp, float]] = {}
    for result in results:
        metric = dict(result["metric"])
        name = metric.pop("__name__", "unknown")
        labels = metric
        if labels:
            label_str = ",".join(
                f'{k}="{v}"' for k, v in sorted(labels.items())
            )
            full_name = f"{name}{{{label_str}}}"
        else:
            full_name = name
        values = result.get("values", [])
        if not values and "value" in result:
            values = [result["value"]]
        point_map: dict[pd.Timestamp, float] = {}
        for ts_str, val_str in values:
            ts = pd.Timestamp.fromtimestamp(float(ts_str), tz="UTC")
            try:
                point_map[ts] = float(val_str)
            except (ValueError, TypeError):
                continue
        if full_name in series_dict:
            series_dict[full_name].update(point_map)
        else:
            series_dict[full_name] = point_map
    records: dict[str, pd.Series] = {}
    for col_name, point_map in series_dict.items():
        s = pd.Series(point_map, dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        records[col_name] = s
    df = pd.DataFrame(records)
    df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index)
    freq = detect_frequency(df.index)
    df = interpolate_with_holes(df, freq)
    metadata = _build_metadata(df.columns, "prometheus", freq, valid_ranges, units)
    _flag_out_of_range(df, metadata)
    return df, metadata


def _parse_influxdb_timestamp(ts_str: str) -> pd.Timestamp | None:
    try:
        ts_val = int(ts_str)
    except ValueError:
        try:
            ts_val = float(ts_str)
        except ValueError:
            return None
    if abs(ts_val) > 1e18:
        ts_ns = ts_val
    elif abs(ts_val) > 1e15:
        ts_ns = ts_val * 1_000
    elif abs(ts_val) > 1e12:
        ts_ns = ts_val * 1_000_000
    else:
        ts_ns = ts_val * 1_000_000_000
    return pd.Timestamp(ts_ns, tz="UTC")


def parse_influxdb_lp(
    content: str,
    valid_ranges: dict[str, tuple[float, float]] | None = None,
    units: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    lines = content.strip().split("\n")
    series_data: dict[str, dict[pd.Timestamp, float]] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first_space = line.find(" ")
        if first_space < 1:
            continue
        measurement_tags = line[:first_space]
        rest = line[first_space + 1 :]
        last_space = rest.rfind(" ")
        if last_space > 0:
            potential_ts = rest[last_space + 1 :]
            timestamp = _parse_influxdb_timestamp(potential_ts)
            if timestamp is not None:
                fields_str = rest[:last_space]
            else:
                fields_str = rest
                timestamp = None
        else:
            fields_str = rest
            timestamp = None
        if "," in measurement_tags:
            parts = measurement_tags.split(",")
            measurement = parts[0]
            tags: dict[str, str] = {}
            for tag_part in parts[1:]:
                if "=" in tag_part:
                    k, v = tag_part.split("=", 1)
                    tags[k] = v
        else:
            measurement = measurement_tags
            tags = {}
        if tags:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            base_name = f"{measurement},{label_str}"
        else:
            base_name = measurement
        fields: dict[str, float] = {}
        for field_part in fields_str.split(","):
            if "=" not in field_part:
                continue
            k, v = field_part.split("=", 1)
            if v.startswith('"') and v.endswith('"'):
                continue
            try:
                fields[k] = float(v.rstrip("i"))
            except ValueError:
                continue
        if not fields or timestamp is None:
            continue
        for field_name, field_value in fields.items():
            if len(fields) > 1:
                col_name = f"{base_name}.{field_name}"
            else:
                col_name = base_name
            if col_name not in series_data:
                series_data[col_name] = {}
            series_data[col_name][timestamp] = field_value
    records: dict[str, pd.Series] = {}
    for col_name, point_map in series_data.items():
        s = pd.Series(point_map, dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        records[col_name] = s
    df = pd.DataFrame(records)
    df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index)
    freq = detect_frequency(df.index)
    df = interpolate_with_holes(df, freq)
    metadata = _build_metadata(df.columns, "influxdb", freq, valid_ranges, units)
    _flag_out_of_range(df, metadata)
    return df, metadata


def shard(
    df: pd.DataFrame, metadata: list[dict]
) -> list[tuple[pd.DataFrame, list[dict]]]:
    if len(df.columns) <= MAX_METRICS and len(df) <= MAX_DATA_POINTS:
        return [(df, metadata)]
    shards: list[tuple[pd.DataFrame, list[dict]]] = []
    for i in range(0, len(df.columns), MAX_METRICS):
        cols = df.columns[i : i + MAX_METRICS]
        sub_df = df[cols]
        sub_meta = [m for m in metadata if m["name"] in cols]
        if len(sub_df) > MAX_DATA_POINTS:
            for j in range(0, len(sub_df), MAX_DATA_POINTS):
                time_shard = sub_df.iloc[j : j + MAX_DATA_POINTS]
                shards.append((time_shard, sub_meta))
        else:
            shards.append((sub_df, sub_meta))
    return shards
