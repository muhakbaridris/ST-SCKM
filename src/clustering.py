"""Clustering evaluation and risk-label helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


RISK_LABELS = ("Low Risk", "Moderate Risk", "High Risk", "Extreme Risk")


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute common internal clustering metrics.

    Noise labels encoded as ``-1`` are excluded from the metric computation.
    """
    labels = np.asarray(labels)
    valid = labels != -1
    unique = np.unique(labels[valid])
    if len(unique) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
    X_valid = X[valid]
    y_valid = labels[valid]
    return {
        "silhouette": float(silhouette_score(X_valid, y_valid)),
        "calinski_harabasz": float(calinski_harabasz_score(X_valid, y_valid)),
        "davies_bouldin": float(davies_bouldin_score(X_valid, y_valid)),
    }


def assign_risk_labels(
    frame: pd.DataFrame,
    label_column: str,
    intensity_column: str = "log_frp",
    risk_labels: tuple[str, ...] = RISK_LABELS,
) -> pd.Series:
    """Map cluster ids to ordered qualitative risk labels.

    Clusters are ranked by the mean value of ``intensity_column``. The lowest
    mean receives the first label and the highest mean receives the last label.
    """
    if label_column not in frame:
        raise KeyError(f"{label_column!r} is not a column in frame")
    if intensity_column not in frame:
        raise KeyError(f"{intensity_column!r} is not a column in frame")

    valid = frame[label_column] != -1
    means = frame.loc[valid].groupby(label_column)[intensity_column].mean().sort_values()
    mapping: dict[int, str] = {-1: "Noise"}
    for idx, cluster_id in enumerate(means.index):
        mapping[int(cluster_id)] = risk_labels[min(idx, len(risk_labels) - 1)]
    return frame[label_column].map(mapping)
