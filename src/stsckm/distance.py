"""Distance functions used by ST-SCKM."""

from __future__ import annotations

import numpy as np


def weighted_spatiotemporal_distance(
    X_spatial: np.ndarray,
    X_temporal: np.ndarray,
    centroids_spatial: np.ndarray,
    centroids_temporal: np.ndarray,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> np.ndarray:
    """Compute squared weighted distances to every cluster centroid.

    Parameters
    ----------
    X_spatial
        Spatial feature matrix with shape ``(n_samples, n_spatial_features)``.
    X_temporal
        Temporal feature matrix with shape ``(n_samples, n_temporal_features)``.
    centroids_spatial
        Spatial centroids with shape ``(n_clusters, n_spatial_features)``.
    centroids_temporal
        Temporal centroids with shape ``(n_clusters, n_temporal_features)``.
    spatial_weight
        Non-negative multiplier for squared spatial distances.
    temporal_weight
        Non-negative multiplier for squared temporal distances.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``(n_samples, n_clusters)``.
    """
    spatial_diff = X_spatial[:, None, :] - centroids_spatial[None, :, :]
    temporal_diff = X_temporal[:, None, :] - centroids_temporal[None, :, :]
    spatial_distance = np.einsum("nkd,nkd->nk", spatial_diff, spatial_diff)
    temporal_distance = np.einsum("nkd,nkd->nk", temporal_diff, temporal_diff)
    return spatial_weight * spatial_distance + temporal_weight * temporal_distance
