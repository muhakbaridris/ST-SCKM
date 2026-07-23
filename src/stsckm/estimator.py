"""ST-SCKM estimator."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .distance import weighted_spatiotemporal_distance
from .graph import knn_indices


class STSCKM(ClusterMixin, BaseEstimator):
    """Spatio-temporal spatially constrained K-means.

    The estimator minimizes weighted spatial and temporal squared distances
    with a soft disagreement penalty on a spatial K-nearest-neighbor graph.
    Assignments are updated sequentially in input row order.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    spatial_weight
        Non-negative weight for squared spatial distance.
    temporal_weight
        Non-negative weight for squared temporal distance.
    lambda_spatial
        Non-negative penalty for each directed neighbor-label disagreement.
    n_neighbors
        Number of spatial neighbors used by the penalty.
    max_iter
        Maximum number of penalized assignment passes.
    tol
        Absolute stopping tolerance for changes in the recorded objective.
    n_init
        Number of K-means initializations.
    random_state
        Seed or random-state instance used for initialization and empty-cluster
        handling.

    Attributes
    ----------
    labels_ : numpy.ndarray of shape (n_samples,)
        Final cluster labels.
    neighbors_ : numpy.ndarray
        Directed KNN indices.
    cluster_centers_spatial_ : numpy.ndarray
        Spatial centroids.
    cluster_centers_temporal_ : numpy.ndarray
        Temporal centroids.
    objective_history_ : list of float
        Penalized objective after each assignment pass.
    objective_ : float
        Final recorded penalized objective.
    n_iter_ : int
        Number of penalized assignment passes.
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
        random_state: int | np.random.RandomState | None = 42,
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

    def fit(self, X_spatial: np.ndarray, X_temporal: np.ndarray) -> STSCKM:
        """Fit ST-SCKM to aligned spatial and temporal matrices.

        Parameters
        ----------
        X_spatial
            Spatial feature matrix.
        X_temporal
            Temporal feature matrix.

        Returns
        -------
        STSCKM
            Fitted estimator.
        """
        self._validate_parameters()
        X_spatial = check_array(X_spatial, dtype=float, ensure_2d=True)
        X_temporal = check_array(X_temporal, dtype=float, ensure_2d=True)
        if len(X_spatial) != len(X_temporal):
            raise ValueError("X_spatial and X_temporal must contain the same number of rows")
        if self.n_clusters > len(X_spatial):
            raise ValueError("n_clusters cannot exceed the number of observations")

        random_state = check_random_state(self.random_state)
        weighted = np.hstack(
            (
                X_spatial * np.sqrt(self.spatial_weight),
                X_temporal * np.sqrt(self.temporal_weight),
            )
        )
        initializer = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_init,
            random_state=random_state,
        ).fit(weighted)
        labels = initializer.labels_.copy()
        neighbors = knn_indices(X_spatial, self.n_neighbors)

        history: list[float] = []
        for _ in range(self.max_iter):
            spatial_centers, temporal_centers = self._compute_centroids(
                X_spatial,
                X_temporal,
                labels,
                random_state,
            )
            distance_cost = weighted_spatiotemporal_distance(
                X_spatial,
                X_temporal,
                spatial_centers,
                temporal_centers,
                self.spatial_weight,
                self.temporal_weight,
            )
            new_labels = self._assign_with_penalty(distance_cost, labels, neighbors)
            objective = self._objective(distance_cost, new_labels, neighbors)
            history.append(objective)

            unchanged = np.array_equal(new_labels, labels)
            small_change = len(history) > 1 and abs(history[-2] - history[-1]) <= self.tol
            labels = new_labels
            if unchanged or small_change:
                break

        self.labels_ = labels
        self.neighbors_ = neighbors
        (
            self.cluster_centers_spatial_,
            self.cluster_centers_temporal_,
        ) = self._compute_centroids(X_spatial, X_temporal, labels, random_state)
        self.objective_history_ = history
        self.objective_ = history[-1]
        self.n_iter_ = len(history)
        self.n_spatial_features_in_ = X_spatial.shape[1]
        self.n_temporal_features_in_ = X_temporal.shape[1]
        return self

    def fit_predict(self, X_spatial: np.ndarray, X_temporal: np.ndarray) -> np.ndarray:
        """Fit the estimator and return final labels."""
        return self.fit(X_spatial, X_temporal).labels_

    def get_objective_history(self) -> np.ndarray:
        """Return a copy of the recorded objective history."""
        check_is_fitted(self, "objective_history_")
        return np.asarray(self.objective_history_, dtype=float).copy()

    def _validate_parameters(self) -> None:
        integer_parameters = {
            "n_clusters": (self.n_clusters, 2),
            "n_neighbors": (self.n_neighbors, 1),
            "max_iter": (self.max_iter, 1),
            "n_init": (self.n_init, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")

        numeric_parameters = {
            "spatial_weight": self.spatial_weight,
            "temporal_weight": self.temporal_weight,
            "lambda_spatial": self.lambda_spatial,
            "tol": self.tol,
        }
        for name, value in numeric_parameters.items():
            if not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.spatial_weight == 0 and self.temporal_weight == 0:
            raise ValueError("spatial_weight and temporal_weight cannot both be zero")

    def _compute_centroids(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        labels: np.ndarray,
        random_state: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        spatial_centers = np.zeros((self.n_clusters, X_spatial.shape[1]), dtype=float)
        temporal_centers = np.zeros((self.n_clusters, X_temporal.shape[1]), dtype=float)
        for cluster in range(self.n_clusters):
            members = labels == cluster
            if np.any(members):
                spatial_centers[cluster] = X_spatial[members].mean(axis=0)
                temporal_centers[cluster] = X_temporal[members].mean(axis=0)
            else:
                replacement = random_state.randint(len(X_spatial))
                spatial_centers[cluster] = X_spatial[replacement]
                temporal_centers[cluster] = X_temporal[replacement]
        return spatial_centers, temporal_centers

    def _assign_with_penalty(
        self,
        distance_cost: np.ndarray,
        labels: np.ndarray,
        neighbors: np.ndarray,
    ) -> np.ndarray:
        new_labels = labels.copy()
        for i in range(distance_cost.shape[0]):
            neighbor_labels = new_labels[neighbors[i]]
            penalty = np.array(
                [np.count_nonzero(neighbor_labels != k) for k in range(self.n_clusters)],
                dtype=float,
            )
            new_labels[i] = int(
                np.argmin(distance_cost[i] + self.lambda_spatial * penalty)
            )
        return new_labels

    def _objective(
        self,
        distance_cost: np.ndarray,
        labels: np.ndarray,
        neighbors: np.ndarray,
    ) -> float:
        within = float(distance_cost[np.arange(len(labels)), labels].sum())
        disagreement = np.count_nonzero(labels[:, None] != labels[neighbors])
        return within + self.lambda_spatial * float(disagreement)
