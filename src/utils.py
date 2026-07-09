"""Data preparation utilities for ST-SCKM examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_sample_wildfire_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a compact California-like wildfire sample dataset."""
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
            "humidity": np.clip(centers[cluster, 3] + rng.normal(0, 9.0, n_samples), 3, 95),
            "wind_speed": np.clip(rng.normal(2.6, 0.9, n_samples), 0.1, None),
            "frp": np.clip(rng.gamma(shape=2.0, scale=centers[cluster, 4] / 2.0), 0.1, None),
        }
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    return frame


def add_default_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add projected, temporal, and log-intensity columns used in examples.

    For lightweight reproducibility, this helper uses a deterministic
    longitude/latitude to meter approximation. For production geodesy, replace
    these two lines with a pyproj projection appropriate to the study area.
    """
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["log_frp"] = np.log1p(out["frp"])
    out["time_days"] = (out["datetime"] - out["datetime"].min()).dt.total_seconds() / 86400.0
    out["x_proj"] = out["longitude"] * 88000.0
    out["y_proj"] = out["latitude"] * 111000.0
    return out


def standardize_features(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Return standardized feature matrix and fitted scaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(frame[columns].to_numpy(dtype=float))
    return X, scaler
