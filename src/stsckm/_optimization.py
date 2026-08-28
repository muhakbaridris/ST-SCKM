"""Shared optimization engine for graph-regularized centroid clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

UpdateScheme = Literal["sequential", "synchronous"]


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable output from the shared optimization engine."""

    labels: np.ndarray
    centers: np.ndarray
    objective_history: tuple[float, ...]
    regularization_adjacency: sparse.csr_matrix


def weighted_feature_distance(
    X: np.ndarray,
    centers: np.ndarray,
    feature_weights: np.ndarray,
) -> np.ndarray:
    """Return weighted squared distances from observations to centers."""
    differences = X[:, None, :] - centers[None, :, :]
    return np.einsum("nkp,p,nkp->nk", differences, feature_weights, differences)


def optimize_graph_regularized_kmeans(
    X: np.ndarray,
    *,
    feature_weights: np.ndarray,
    adjacency: sparse.csr_matrix,
    n_clusters: int,
    graph_penalty: float,
    max_iter: int,
    tol: float,
    n_init: int,
    random_state: np.random.RandomState,
    update_scheme: UpdateScheme,
) -> OptimizationResult:
    """Fit weighted K-means with a soft graph-disagreement penalty."""
    regularization_graph = ((adjacency + adjacency.T) * 0.5).tocsr()
    regularization_graph.eliminate_zeros()
    regularization_graph.sort_indices()

    initializer = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
    ).fit(X * np.sqrt(feature_weights))
    labels = initializer.labels_.copy()

    history: list[float] = []
    for _ in range(max_iter):
        centers = _compute_centers(X, labels, n_clusters, random_state)
        distance_cost = weighted_feature_distance(X, centers, feature_weights)
        new_labels = _assign_with_penalty(
            distance_cost,
            labels,
            regularization_graph,
            n_clusters=n_clusters,
            graph_penalty=graph_penalty,
            update_scheme=update_scheme,
        )
        updated_centers = _compute_centers(X, new_labels, n_clusters, random_state)
        updated_cost = weighted_feature_distance(X, updated_centers, feature_weights)
        objective = _objective(
            updated_cost,
            new_labels,
            regularization_graph,
            graph_penalty=graph_penalty,
        )
        history.append(objective)

        unchanged = np.array_equal(new_labels, labels)
        small_change = len(history) > 1 and abs(history[-2] - history[-1]) <= tol
        labels = new_labels
        if unchanged or small_change:
            break

    centers = _compute_centers(X, labels, n_clusters, random_state)
    return OptimizationResult(
        labels=labels,
        centers=centers,
        objective_history=tuple(history),
        regularization_adjacency=regularization_graph,
    )


def _compute_centers(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    centers = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for cluster in range(n_clusters):
        members = labels == cluster
        if np.any(members):
            centers[cluster] = X[members].mean(axis=0)
        else:
            centers[cluster] = X[random_state.randint(len(X))]
    return centers


def _assign_with_penalty(
    distance_cost: np.ndarray,
    labels: np.ndarray,
    graph: sparse.csr_matrix,
    *,
    n_clusters: int,
    graph_penalty: float,
    update_scheme: UpdateScheme,
) -> np.ndarray:
    new_labels = labels.copy()
    reference = labels if update_scheme == "synchronous" else new_labels
    for index in range(distance_cost.shape[0]):
        start, stop = graph.indptr[index], graph.indptr[index + 1]
        neighbors = graph.indices[start:stop]
        weights = graph.data[start:stop]
        neighbor_labels = reference[neighbors]
        penalty = np.asarray(
            [weights[neighbor_labels != cluster].sum() for cluster in range(n_clusters)],
            dtype=float,
        )
        new_labels[index] = int(np.argmin(distance_cost[index] + graph_penalty * penalty))
    return new_labels


def _objective(
    distance_cost: np.ndarray,
    labels: np.ndarray,
    graph: sparse.csr_matrix,
    *,
    graph_penalty: float,
) -> float:
    within = float(distance_cost[np.arange(len(labels)), labels].sum())
    rows, columns = graph.nonzero()
    disagreement = float(graph.data[labels[rows] != labels[columns]].sum())
    return within + 0.5 * graph_penalty * disagreement
