"""Data preparation helpers and a compact synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_sample_wildfire_data(
    n_samples: int = 1000,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Generate a California-like synthetic wildfire event dataset."""
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(random_state)
    centers = np.array(
        [
            [41.0, -123.2, 280.0, 65.0, 45.0],
            [38.4, -121.5, 289.0, 42.0, 70.0],
            [36.5, -119.3, 296.0, 29.0, 95.0],
            [34.1, -117.2, 298.0, 25.0, 150.0],
        ]
    )
    cluster = rng.integers(0, len(centers), n_samples)
    dates = pd.date_range("2019-01-01", "2024-12-31", periods=n_samples)
    frame = pd.DataFrame(
        {
            "latitude": centers[cluster, 0] + rng.normal(0, 0.55, n_samples),
            "longitude": centers[cluster, 1] + rng.normal(0, 0.55, n_samples),
            "datetime": rng.choice(dates, n_samples, replace=True),
            "temperature": centers[cluster, 2] + rng.normal(0, 4.5, n_samples),
            "humidity": np.clip(
                centers[cluster, 3] + rng.normal(0, 9.0, n_samples),
                3,
                95,
            ),
            "wind_speed": np.clip(rng.normal(2.6, 0.9, n_samples), 0.1, None),
            "frp": np.clip(
                rng.gamma(shape=2.0, scale=centers[cluster, 4] / 2.0),
                0.1,
                None,
            ),
        }
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values("datetime").reset_index(drop=True)


def add_default_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight projected, temporal, and log-intensity columns.

    This deterministic conversion is intended for the bundled example. Use a
    suitable projected coordinate reference system for scientific geodesy.
    """
    required = {"longitude", "latitude", "datetime", "frp"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    result["log_frp"] = np.log1p(result["frp"].astype(float))
    elapsed = result["datetime"] - result["datetime"].min()
    result["time_days"] = elapsed.dt.total_seconds() / 86400.0
    result["x_proj"] = result["longitude"].astype(float) * 88000.0
    result["y_proj"] = result["latitude"].astype(float) * 111000.0
    return result


def standardize_features(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, StandardScaler]:
    """Standardize selected columns and return the fitted scaler."""
    if not columns:
        raise ValueError("columns cannot be empty")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"missing requested columns: {sorted(missing)}")
    scaler = StandardScaler()
    values = scaler.fit_transform(frame[list(columns)].to_numpy(dtype=float))
    return values, scaler
