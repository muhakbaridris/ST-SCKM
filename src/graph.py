"""Spatial neighborhood graph utilities."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_indices(X_spatial: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Build a KNN index array from spatial coordinates.

    The returned array has shape ``(n_samples, n_neighbors)`` and excludes each
    observation itself from its own neighbor list.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if len(X_spatial) <= 1:
        raise ValueError("at least two observations are required")

    k = min(n_neighbors + 1, len(X_spatial))
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X_spatial)
    indices = nbrs.kneighbors(X_spatial, return_distance=False)

    cleaned = []
    for i, row in enumerate(indices):
        row = [j for j in row if j != i]
        cleaned.append(row[: min(n_neighbors, len(row))])
    return np.asarray(cleaned, dtype=int)
