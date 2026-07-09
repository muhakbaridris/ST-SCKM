"""Spatio-Temporal Spatially Constrained K-Means estimator."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array

from .distance import weighted_spatiotemporal_distance
from .graph import knn_indices


class STSCKM:
    """Spatio-Temporal Spatially Constrained K-Means.

    The estimator minimizes a weighted spatio-temporal k-means objective with a
    soft KNN penalty. During assignment, each point pays a penalty when its
    candidate cluster differs from the current labels of its spatial neighbors.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        spatial_weight: float = 0.5,
        temporal_weight: float = 1.5,
        lambda_spatial: float = 1.0,
        n_neighbors: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int | None = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.lambda_spatial = lambda_spatial
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X_spatial: np.ndarray, X_temporal: np.ndarray) -> "STSCKM":
        """Fit ST-SCKM using spatial and temporal feature matrices."""
        X_spatial = check_array(X_spatial, dtype=float)
        X_temporal = check_array(X_temporal, dtype=float)
        if len(X_spatial) != len(X_temporal):
            raise ValueError("X_spatial and X_temporal must contain the same number of rows")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if self.n_clusters > len(X_spatial):
            raise ValueError("n_clusters cannot exceed the number of observations")

        X_weighted = np.hstack(
            [
                X_spatial * np.sqrt(self.spatial_weight),
                X_temporal * np.sqrt(self.temporal_weight),
            ]
        )
        init = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(X_weighted)
        labels = init.labels_.copy()
        neighbors = knn_indices(X_spatial, self.n_neighbors)

        objective_history: list[float] = []
        for _ in range(self.max_iter):
            centroids_spatial, centroids_temporal = self._compute_centroids(
                X_spatial, X_temporal, labels
            )
            distance_cost = weighted_spatiotemporal_distance(
                X_spatial,
                X_temporal,
                centroids_spatial,
                centroids_temporal,
                self.spatial_weight,
                self.temporal_weight,
            )
            new_labels = self._assign_with_penalty(distance_cost, labels, neighbors)
            objective = self._objective(distance_cost, new_labels, neighbors)
            objective_history.append(objective)

            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            if len(objective_history) > 1:
                improvement = objective_history[-2] - objective_history[-1]
                if abs(improvement) <= self.tol:
                    labels = new_labels
                    break
            labels = new_labels

        self.labels_ = labels
        self.neighbors_ = neighbors
        self.cluster_centers_spatial_, self.cluster_centers_temporal_ = self._compute_centroids(
            X_spatial, X_temporal, labels
        )
        self.objective_history_ = objective_history
        self.n_iter_ = len(objective_history)
        return self

    def fit_predict(self, X_spatial: np.ndarray, X_temporal: np.ndarray) -> np.ndarray:
        """Fit the model and return labels."""
        return self.fit(X_spatial, X_temporal).labels_

    def _compute_centroids(
        self, X_spatial: np.ndarray, X_temporal: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        spatial_centers = np.zeros((self.n_clusters, X_spatial.shape[1]), dtype=float)
        temporal_centers = np.zeros((self.n_clusters, X_temporal.shape[1]), dtype=float)
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                spatial_centers[k] = X_spatial[mask].mean(axis=0)
                temporal_centers[k] = X_temporal[mask].mean(axis=0)
            else:
                spatial_centers[k] = X_spatial[np.random.randint(0, len(X_spatial))]
                temporal_centers[k] = X_temporal[np.random.randint(0, len(X_temporal))]
        return spatial_centers, temporal_centers

    def _assign_with_penalty(
        self, distance_cost: np.ndarray, labels: np.ndarray, neighbors: np.ndarray
    ) -> np.ndarray:
        new_labels = labels.copy()
        for i in range(distance_cost.shape[0]):
            neighbor_labels = new_labels[neighbors[i]]
            penalty = np.array(
                [np.sum(neighbor_labels != k) for k in range(self.n_clusters)], dtype=float
            )
            new_labels[i] = int(np.argmin(distance_cost[i] + self.lambda_spatial * penalty))
        return new_labels

    def _objective(
        self, distance_cost: np.ndarray, labels: np.ndarray, neighbors: np.ndarray
    ) -> float:
        inertia = float(distance_cost[np.arange(len(labels)), labels].sum())
        penalty = 0.0
        for i, row in enumerate(neighbors):
            penalty += np.sum(labels[row] != labels[i])
        return inertia + self.lambda_spatial * float(penalty)
