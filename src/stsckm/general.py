"""General-purpose graph-regularized K-means estimator."""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from ._optimization import (
    UpdateScheme,
    optimize_graph_regularized_kmeans,
    weighted_feature_distance,
)
from .diagnostics import graph_diagnostics
from .graph import adjacency_to_neighbors, spatial_graph, validate_adjacency


class GraphRegularizedKMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Cluster a feature matrix with a soft graph-disagreement penalty.

    This estimator is the domain-general counterpart of :class:`STSCKM`.
    Centroid fit is computed from ``X`` while graph construction may use a
    separate representation passed as ``graph_features``. A custom dense or
    sparse adjacency matrix can be supplied instead.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    graph_penalty
        Non-negative multiplier for weighted neighbor-label disagreement.
    feature_weights
        Optional non-negative weight for each column of ``X``. A scalar is
        broadcast to all columns. By default every column has weight one.
    graph_type
        K-nearest-neighbor or radius graph when ``adjacency`` is not supplied.
    n_neighbors, radius, graph_symmetrize
        Controls for the automatically constructed graph.
    update_scheme
        Sequential or synchronous label updates.
    max_iter, tol, n_init, random_state
        Optimization and initialization controls.

    Notes
    -----
    :meth:`predict` assigns observations by centroid distance alone because a
    graph connecting new observations to the fitted sample is not defined.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        graph_penalty: float = 1.0,
        feature_weights: float | tuple[float, ...] | list[float] | None = None,
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
        self.graph_penalty = graph_penalty
        self.feature_weights = feature_weights
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
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        adjacency: np.ndarray | sparse.spmatrix | None = None,
        graph_features: np.ndarray | None = None,
    ) -> GraphRegularizedKMeans:
        """Fit the estimator.

        ``y`` is accepted and ignored for compatibility with scikit-learn
        pipelines.
        """
        del y
        self._validate_parameters()
        original_columns = getattr(X, "columns", None)
        X = check_array(X, dtype=float, ensure_2d=True)
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of observations")
        feature_weights = self._resolve_feature_weights(X.shape[1])
        graph = self._resolve_graph(X, adjacency, graph_features)
        result = optimize_graph_regularized_kmeans(
            X,
            feature_weights=feature_weights,
            adjacency=graph,
            n_clusters=int(self.n_clusters),
            graph_penalty=float(self.graph_penalty),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            n_init=int(self.n_init),
            random_state=check_random_state(self.random_state),
            update_scheme=self.update_scheme,
        )

        self.labels_ = result.labels
        self.cluster_centers_ = result.centers
        self.feature_weights_ = feature_weights
        self.adjacency_ = graph
        self.regularization_adjacency_ = result.regularization_adjacency
        self.neighbor_indices_ = adjacency_to_neighbors(graph)
        self.neighbors_ = self._dense_neighbors_if_regular(self.neighbor_indices_)
        self.objective_history_ = list(result.objective_history)
        self.objective_ = self.objective_history_[-1]
        self.n_iter_ = len(self.objective_history_)
        self.n_features_in_ = X.shape[1]
        if original_columns is not None and all(isinstance(name, str) for name in original_columns):
            self.feature_names_in_ = np.asarray(original_columns, dtype=object)
        self.graph_type_ = "custom" if adjacency is not None else self.graph_type
        self.graph_diagnostics_ = graph_diagnostics(self.labels_, graph)
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        adjacency: np.ndarray | sparse.spmatrix | None = None,
        graph_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit the estimator and return final labels."""
        return self.fit(
            X,
            y,
            adjacency=adjacency,
            graph_features=graph_features,
        ).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return weighted squared distances to fitted centroids."""
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X, dtype=float, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from the fitted data")
        return weighted_feature_distance(X, self.cluster_centers_, self.feature_weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign observations to their nearest centroid without graph costs."""
        return np.argmin(self.transform(X), axis=1)

    def get_objective_history(self) -> np.ndarray:
        """Return a copy of the recorded objective history."""
        check_is_fitted(self, "objective_history_")
        return np.asarray(self.objective_history_, dtype=float).copy()

    def _resolve_feature_weights(self, n_features: int) -> np.ndarray:
        if self.feature_weights is None:
            weights = np.ones(n_features, dtype=float)
        elif isinstance(self.feature_weights, numbers.Real):
            weights = np.full(n_features, float(self.feature_weights), dtype=float)
        else:
            weights = np.asarray(self.feature_weights, dtype=float)
            if weights.ndim != 1 or len(weights) != n_features:
                raise ValueError("feature_weights must have one entry per feature")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("feature_weights must be finite and non-negative")
        if not np.any(weights > 0):
            raise ValueError("at least one feature weight must be positive")
        return weights

    def _resolve_graph(
        self,
        X: np.ndarray,
        adjacency: np.ndarray | sparse.spmatrix | None,
        graph_features: np.ndarray | None,
    ) -> sparse.csr_matrix:
        if adjacency is not None and graph_features is not None:
            raise ValueError("pass adjacency or graph_features, not both")
        if adjacency is not None:
            return validate_adjacency(
                adjacency,
                n_samples=len(X),
                symmetrize=self.graph_symmetrize,
            )
        graph_input = X if graph_features is None else check_array(
            graph_features,
            dtype=float,
            ensure_2d=True,
        )
        if len(graph_input) != len(X):
            raise ValueError("graph_features and X must contain the same number of rows")
        return spatial_graph(
            graph_input,
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
        for name, value in {"graph_penalty": self.graph_penalty, "tol": self.tol}.items():
            if not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
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

    @staticmethod
    def _dense_neighbors_if_regular(rows: list[np.ndarray]) -> np.ndarray | None:
        lengths = {len(row) for row in rows}
        if len(lengths) == 1 and rows:
            return np.vstack(rows).astype(int, copy=False)
        return None
