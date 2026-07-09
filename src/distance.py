"""Distance helpers for spatio-temporal clustering."""

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
    """Return squared weighted distance from observations to centroids.

    Parameters
    ----------
    X_spatial:
        Spatial feature matrix with shape ``(n_samples, n_spatial_features)``.
    X_temporal:
        Temporal feature matrix with shape ``(n_samples, n_temporal_features)``.
    centroids_spatial:
        Spatial centroids with shape ``(n_clusters, n_spatial_features)``.
    centroids_temporal:
        Temporal centroids with shape ``(n_clusters, n_temporal_features)``.
    spatial_weight, temporal_weight:
        Non-negative weights applied to squared spatial and temporal distances.
    """
    spatial_diff = X_spatial[:, None, :] - centroids_spatial[None, :, :]
    temporal_diff = X_temporal[:, None, :] - centroids_temporal[None, :, :]
    spatial_dist = np.sum(spatial_diff * spatial_diff, axis=2)
    temporal_dist = np.sum(temporal_diff * temporal_diff, axis=2)
    return spatial_weight * spatial_dist + temporal_weight * temporal_dist
