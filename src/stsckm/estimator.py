"""Graph-regularized spatio-temporal K-means estimator."""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .diagnostics import graph_diagnostics
from .distance import weighted_spatiotemporal_distance
from .graph import adjacency_to_neighbors, spatial_graph, validate_adjacency

UpdateScheme = Literal["sequential", "synchronous"]


class STSCKM(ClusterMixin, TransformerMixin, BaseEstimator):
    """Graph-regularized spatio-temporal K-means.

    The estimator combines weighted spatial and temporal squared distances
    with a soft disagreement penalty on a sparse neighborhood graph. A graph
    can be constructed from K-nearest neighbors or a radius, or supplied by
    the caller as a dense or sparse adjacency matrix.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    spatial_weight, temporal_weight
        Non-negative weights applied to spatial and temporal squared distance.
    lambda_spatial
        Non-negative graph disagreement penalty.
    graph_type
        Graph constructed when ``adjacency`` is not passed to :meth:`fit`.
    n_neighbors
        Number of neighbors for ``graph_type="knn"``.
    radius
        Positive threshold required by ``graph_type="radius"``.
    graph_symmetrize
        Keep directed edges, take their union, or retain mutual edges.
    update_scheme
        Sequential Gauss-Seidel-like updates or synchronous label updates.
    max_iter, tol, n_init, random_state
        Optimization and initialization controls.
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
        graph_type: Literal["knn", "radius"] = "knn",
        radius: float | None = None,
        graph_symmetrize: Literal["none", "union", "mutual"] = "none",
        update_scheme: UpdateScheme = "sequential",
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
        self.graph_type = graph_type
        self.radius = radius
        self.graph_symmetrize = graph_symmetrize
        self.update_scheme = update_scheme

    def fit(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        *,
        adjacency: np.ndarray | sparse.spmatrix | None = None,
    ) -> STSCKM:
        """Fit the estimator to aligned spatial and temporal matrices."""
        self._validate_parameters()
        X_spatial, X_temporal = self._validate_inputs(X_spatial, X_temporal)
        random_state = check_random_state(self.random_state)
        graph = self._resolve_graph(X_spatial, adjacency)
        regularization_graph = ((graph + graph.T) * 0.5).tocsr()
        regularization_graph.eliminate_zeros()

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
            new_labels = self._assign_with_penalty(
                distance_cost,
                labels,
                regularization_graph,
            )
            new_spatial_centers, new_temporal_centers = self._compute_centroids(
                X_spatial,
                X_temporal,
                new_labels,
                random_state,
            )
            updated_distance_cost = weighted_spatiotemporal_distance(
                X_spatial,
                X_temporal,
                new_spatial_centers,
                new_temporal_centers,
                self.spatial_weight,
                self.temporal_weight,
            )
            objective = self._objective(
                updated_distance_cost,
                new_labels,
                regularization_graph,
            )
            history.append(objective)

            unchanged = np.array_equal(new_labels, labels)
            small_change = len(history) > 1 and abs(history[-2] - history[-1]) <= self.tol
            labels = new_labels
            if unchanged or small_change:
                break

        self.labels_ = labels
        self.adjacency_ = graph
        self.regularization_adjacency_ = regularization_graph
        self.neighbor_indices_ = adjacency_to_neighbors(graph)
        self.neighbors_ = self._dense_neighbors_if_regular(self.neighbor_indices_)
        (
            self.cluster_centers_spatial_,
            self.cluster_centers_temporal_,
        ) = self._compute_centroids(X_spatial, X_temporal, labels, random_state)
        self.objective_history_ = history
        self.objective_ = history[-1]
        self.n_iter_ = len(history)
        self.n_spatial_features_in_ = X_spatial.shape[1]
        self.n_temporal_features_in_ = X_temporal.shape[1]
        self.graph_type_ = "custom" if adjacency is not None else self.graph_type
        self.graph_diagnostics_ = graph_diagnostics(labels, graph)
        return self

    def fit_predict(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
        *,
        adjacency: np.ndarray | sparse.spmatrix | None = None,
    ) -> np.ndarray:
        """Fit the estimator and return final labels."""
        return self.fit(X_spatial, X_temporal, adjacency=adjacency).labels_

    def transform(self, X_spatial: np.ndarray, X_temporal: np.ndarray) -> np.ndarray:
        """Return weighted distances to the fitted cluster centroids.

        Graph penalties are not included because a graph for new observations
        is not defined by the fitted object.
        """
        check_is_fitted(self, "cluster_centers_spatial_")
        X_spatial = check_array(X_spatial, dtype=float, ensure_2d=True)
        X_temporal = check_array(X_temporal, dtype=float, ensure_2d=True)
        if len(X_spatial) != len(X_temporal):
            raise ValueError("X_spatial and X_temporal must contain the same number of rows")
        if X_spatial.shape[1] != self.n_spatial_features_in_:
            raise ValueError("X_spatial has a different number of features from the fitted data")
        if X_temporal.shape[1] != self.n_temporal_features_in_:
            raise ValueError("X_temporal has a different number of features from the fitted data")
        return weighted_spatiotemporal_distance(
            X_spatial,
            X_temporal,
            self.cluster_centers_spatial_,
            self.cluster_centers_temporal_,
            self.spatial_weight,
            self.temporal_weight,
        )

    def get_objective_history(self) -> np.ndarray:
        """Return a copy of the recorded objective history."""
        check_is_fitted(self, "objective_history_")
        return np.asarray(self.objective_history_, dtype=float).copy()

    def _validate_inputs(
        self,
        X_spatial: np.ndarray,
        X_temporal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_spatial = check_array(X_spatial, dtype=float, ensure_2d=True)
        X_temporal = check_array(X_temporal, dtype=float, ensure_2d=True)
        if len(X_spatial) != len(X_temporal):
            raise ValueError("X_spatial and X_temporal must contain the same number of rows")
        if self.n_clusters > len(X_spatial):
            raise ValueError("n_clusters cannot exceed the number of observations")
        return X_spatial, X_temporal

    def _resolve_graph(
        self,
        X_spatial: np.ndarray,
        adjacency: np.ndarray | sparse.spmatrix | None,
    ) -> sparse.csr_matrix:
        if adjacency is not None:
            return validate_adjacency(
                adjacency,
                n_samples=len(X_spatial),
                symmetrize=self.graph_symmetrize,
            )
        return spatial_graph(
            X_spatial,
            graph_type=self.graph_type,
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            symmetrize=self.graph_symmetrize,
        )

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
        if self.graph_type not in {"knn", "radius"}:
            raise ValueError("graph_type must be 'knn' or 'radius'")
        if self.graph_symmetrize not in {"none", "union", "mutual"}:
            raise ValueError("graph_symmetrize must be 'none', 'union', or 'mutual'")
        if self.update_scheme not in {"sequential", "synchronous"}:
            raise ValueError("update_scheme must be 'sequential' or 'synchronous'")
        if self.graph_type == "radius" and (
            not isinstance(self.radius, numbers.Real)
            or not np.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise ValueError("radius must be a finite positive number for a radius graph")

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
        graph: sparse.csr_matrix,
    ) -> np.ndarray:
        new_labels = labels.copy()
        reference = labels if self.update_scheme == "synchronous" else new_labels
        for index in range(distance_cost.shape[0]):
            start, stop = graph.indptr[index], graph.indptr[index + 1]
            neighbors = graph.indices[start:stop]
            weights = graph.data[start:stop]
            neighbor_labels = reference[neighbors]
            penalty = np.asarray(
                [weights[neighbor_labels != cluster].sum() for cluster in range(self.n_clusters)],
                dtype=float,
            )
            new_labels[index] = int(
                np.argmin(distance_cost[index] + self.lambda_spatial * penalty)
            )
        return new_labels

    def _objective(
        self,
        distance_cost: np.ndarray,
        labels: np.ndarray,
        graph: sparse.csr_matrix,
    ) -> float:
        within = float(distance_cost[np.arange(len(labels)), labels].sum())
        rows, columns = graph.nonzero()
        disagreement = float(graph.data[labels[rows] != labels[columns]].sum())
        return within + 0.5 * self.lambda_spatial * disagreement

    @staticmethod
    def _dense_neighbors_if_regular(rows: list[np.ndarray]) -> np.ndarray | None:
        lengths = {len(row) for row in rows}
        if len(lengths) == 1 and rows:
            return np.vstack(rows).astype(int, copy=False)
        return None
