"""Fetch and clean NOAA Global Hourly observations, and prepare them for modeling."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

NOAA_GLOBAL_HOURLY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_global_hourly(
    station: str,
    start_date: str,
    end_date: str,
    *,
    endpoint: str = NOAA_GLOBAL_HOURLY_URL,
) -> pd.DataFrame:
    """Fetch NOAA Global Hourly records for one station and date range.

    Dates use ISO format (for example, ``2024-01-01``). The API returns
    JSON records; an empty result is returned as an empty DataFrame.
    """
    query = urlencode(
        {
            "dataset": "global-hourly",
            "stations": station,
            "startDate": start_date,
            "endDate": end_date,
            "format": "json",
            "units": "metric",
            "includeAttributes": "false",
        }
    )
    request = Request(f"{endpoint}?{query}", headers={"User-Agent": "ExaGEOhackathon/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)

    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


def _packed_number(value: object, scale: float = 1.0) -> float:
    """Read a packed NOAA value and turn the missing-value marker into NaN."""
    if pd.isna(value):
        return float("nan")
    text = str(value).split(",", 1)[0].strip()
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    if abs(number) >= 9990:
        return float("nan")
    return number * scale


def clean_global_hourly(records: pd.DataFrame) -> pd.DataFrame:
    """Clean common Global Hourly fields into analysis-ready columns."""
    if records.empty:
        return records.copy()

    cleaned = records.copy()
    cleaned.columns = [str(column).upper() for column in cleaned.columns]
    cleaned["DATE"] = pd.to_datetime(cleaned["DATE"], utc=True, errors="coerce")
    cleaned["temperature_c"] = cleaned.get("TMP", pd.Series(index=cleaned.index)).map(
        lambda value: _packed_number(value, 0.1)
    )
    cleaned["dew_point_c"] = cleaned.get("DEW", pd.Series(index=cleaned.index)).map(
        lambda value: _packed_number(value, 0.1)
    )
    cleaned["visibility_km"] = cleaned.get("VIS", pd.Series(index=cleaned.index)).map(
        lambda value: _packed_number(value, 0.1)
    )

    wind_parts = cleaned.get("WND", pd.Series(index=cleaned.index)).fillna("").astype(str).str.split(",")
    cleaned["wind_direction_deg"] = pd.to_numeric(wind_parts.str[0], errors="coerce").replace(999, float("nan"))
    cleaned["wind_type"] = wind_parts.str[1]
    cleaned["wind_speed_mps"] = pd.to_numeric(wind_parts.str[3], errors="coerce") / 10
    cleaned.loc[cleaned["wind_speed_mps"] >= 999, "wind_speed_mps"] = float("nan")

    return (
        cleaned.dropna(subset=["DATE"])
        .drop_duplicates(subset=["DATE"])
        .sort_values("DATE")
        .reset_index(drop=True)
    )


def make_supervised_sequences(
    values: np.ndarray | pd.Series,
    *,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1D series into sliding-window ``(X, y)`` pairs for sequence models.

    ``X`` has shape ``(samples, lookback, 1)`` and ``y`` has shape ``(samples, horizon)``.
    """
    series = np.asarray(values, dtype="float32").reshape(-1)
    n_samples = len(series) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("Not enough observations for the requested lookback/horizon.")

    X = np.stack([series[i : i + lookback] for i in range(n_samples)])
    y = np.stack([series[i + lookback : i + lookback + horizon] for i in range(n_samples)])
    return X[..., np.newaxis], y
