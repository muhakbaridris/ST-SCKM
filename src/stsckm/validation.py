"""Cluster evaluation and post hoc profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.utils.validation import check_array

RISK_LABELS = ("Low Risk", "Moderate Risk", "High Risk", "Extreme Risk")


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute common internal clustering indices.

    Observations labeled ``-1`` are treated as noise and excluded.
    """
    X = check_array(X, dtype=float, ensure_2d=True)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("labels must be one-dimensional and aligned with X")
    valid = labels != -1
    unique = np.unique(labels[valid])
    if len(unique) < 2 or valid.sum() <= len(unique):
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    X_valid = X[valid]
    labels_valid = labels[valid]
    return {
        "silhouette": float(silhouette_score(X_valid, labels_valid)),
        "calinski_harabasz": float(
            calinski_harabasz_score(X_valid, labels_valid)
        ),
        "davies_bouldin": float(davies_bouldin_score(X_valid, labels_valid)),
    }


def neighbor_disagreement(labels: np.ndarray, neighbors: np.ndarray) -> float:
    """Return the fraction of directed neighbor edges with different labels."""
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbors)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if neighbors.ndim != 2 or len(neighbors) != len(labels):
        raise ValueError("neighbors must be a two-dimensional array aligned with labels")
    if neighbors.size == 0:
        return float("nan")
    if neighbors.min() < 0 or neighbors.max() >= len(labels):
        raise ValueError("neighbors contains an out-of-range index")
    return float(np.mean(labels[:, None] != labels[neighbors]))


def assign_risk_labels(
    frame: pd.DataFrame,
    label_column: str,
    intensity_column: str = "log_frp",
    risk_labels: tuple[str, ...] = RISK_LABELS,
) -> pd.Series:
    """Order clusters by mean intensity and assign qualitative labels."""
    if label_column not in frame:
        raise KeyError(f"{label_column!r} is not a column in frame")
    if intensity_column not in frame:
        raise KeyError(f"{intensity_column!r} is not a column in frame")
    if not risk_labels:
        raise ValueError("risk_labels cannot be empty")

    valid = frame[label_column] != -1
    means = (
        frame.loc[valid]
        .groupby(label_column, observed=False)[intensity_column]
        .mean()
        .sort_values()
    )
    mapping: dict[int, str] = {-1: "Noise"}
    for index, cluster_id in enumerate(means.index):
        mapping[int(cluster_id)] = risk_labels[min(index, len(risk_labels) - 1)]
    return frame[label_column].map(mapping)
