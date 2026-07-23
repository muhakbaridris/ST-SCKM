"""Spatial neighborhood graph utilities."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array


def knn_indices(X_spatial: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Return directed spatial K-nearest-neighbor indices.

    Each observation is excluded from its own neighbor list. If
    ``n_neighbors`` exceeds ``n_samples - 1``, all other observations are
    returned.

    Parameters
    ----------
    X_spatial
        Spatial feature matrix.
    n_neighbors
        Requested number of neighbors for each observation.

    Returns
    -------
    numpy.ndarray
        Integer array with shape ``(n_samples, effective_n_neighbors)``.
    """
    X_spatial = check_array(X_spatial, dtype=float, ensure_2d=True)
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be an integer greater than or equal to 1")
    n_samples = len(X_spatial)
    if n_samples < 2:
        raise ValueError("at least two observations are required")

    effective = min(int(n_neighbors), n_samples - 1)
    search = NearestNeighbors(n_neighbors=effective + 1)
    search.fit(X_spatial)
    raw = search.kneighbors(X_spatial, return_distance=False)

    neighbors = np.empty((n_samples, effective), dtype=int)
    for i, row in enumerate(raw):
        cleaned = row[row != i]
        neighbors[i] = cleaned[:effective]
    return neighbors
