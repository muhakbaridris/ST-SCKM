"""Sparse spatial neighborhood graph construction and validation."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.utils.validation import check_array

GraphType = Literal["knn", "radius"]
SymmetrizeMode = Literal["none", "union", "mutual"]
NormalizeMode = Literal["none", "max", "sum"]


def validate_adjacency(
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    n_samples: int | None = None,
    symmetrize: SymmetrizeMode = "none",
) -> sparse.csr_matrix:
    """Validate and return a non-negative CSR adjacency matrix.

    Diagonal entries are removed. Edge weights are retained and enter the
    disagreement penalty multiplicatively.
    """
    if symmetrize not in {"none", "union", "mutual"}:
        raise ValueError("symmetrize must be 'none', 'union', or 'mutual'")
    if sparse.issparse(adjacency):
        graph = sparse.csr_matrix(adjacency, dtype=float, copy=True)
    else:
        values = np.asarray(adjacency, dtype=float)
        if values.ndim != 2:
            raise ValueError("adjacency must be a two-dimensional square matrix")
        graph = sparse.csr_matrix(values)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    if n_samples is not None and graph.shape != (n_samples, n_samples):
        raise ValueError("adjacency shape must match the number of observations")
    if graph.data.size and (
        not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0)
    ):
        raise ValueError("adjacency weights must be finite and non-negative")

    graph.setdiag(0)
    graph.eliminate_zeros()
    if symmetrize == "union":
        graph = graph.maximum(graph.T).tocsr()
    elif symmetrize == "mutual":
        graph = graph.minimum(graph.T).tocsr()
    graph.sort_indices()
    return graph


def spatial_graph(
    X_spatial: np.ndarray,
    *,
    graph_type: GraphType = "knn",
    n_neighbors: int = 5,
    radius: float | None = None,
    symmetrize: SymmetrizeMode = "none",
) -> sparse.csr_matrix:
    """Construct a sparse spatial graph from coordinates or embeddings."""
    X_spatial = check_array(X_spatial, dtype=float, ensure_2d=True)
    n_samples = len(X_spatial)
    if n_samples < 2:
        raise ValueError("at least two observations are required")
    if graph_type not in {"knn", "radius"}:
        raise ValueError("graph_type must be 'knn' or 'radius'")

    if graph_type == "knn":
        if not isinstance(n_neighbors, numbers.Integral) or n_neighbors < 1:
            raise ValueError("n_neighbors must be an integer greater than or equal to 1")
        effective = min(int(n_neighbors), n_samples - 1)
        graph = kneighbors_graph(
            X_spatial,
            n_neighbors=effective,
            mode="connectivity",
            include_self=False,
        )
    else:
        if not isinstance(radius, numbers.Real) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be a finite positive number for a radius graph")
        graph = radius_neighbors_graph(
            X_spatial,
            radius=float(radius),
            mode="connectivity",
            include_self=False,
        )
    return validate_adjacency(graph, n_samples=n_samples, symmetrize=symmetrize)


def adjacency_to_neighbors(adjacency: np.ndarray | sparse.spmatrix) -> list[np.ndarray]:
    """Return one integer neighbor array per adjacency row."""
    graph = validate_adjacency(adjacency)
    return [
        graph.indices[graph.indptr[index] : graph.indptr[index + 1]].copy()
        for index in range(graph.shape[0])
    ]


def combine_adjacencies(
    adjacencies: Sequence[np.ndarray | sparse.spmatrix],
    *,
    weights: Sequence[float] | None = None,
    normalize: NormalizeMode = "none",
    symmetrize: SymmetrizeMode = "none",
) -> sparse.csr_matrix:
    """Combine multiple graph layers into one weighted sparse adjacency.

    Parameters
    ----------
    adjacencies
        Non-empty sequence of square adjacency matrices with identical shape.
    weights
        Optional non-negative multiplier for each graph layer.
    normalize
        Normalize each layer by its maximum edge weight, total edge weight, or
        not at all before applying ``weights``.
    symmetrize
        Optional symmetrization applied after the layers are combined.

    Notes
    -----
    Layer normalization changes the interpretation of the graph penalty. The
    returned matrix records the exact combined weights used by the estimator.
    """
    if not adjacencies:
        raise ValueError("adjacencies cannot be empty")
    if normalize not in {"none", "max", "sum"}:
        raise ValueError("normalize must be 'none', 'max', or 'sum'")
    if weights is None:
        layer_weights = np.ones(len(adjacencies), dtype=float)
    else:
        layer_weights = np.asarray(weights, dtype=float)
        if layer_weights.ndim != 1 or len(layer_weights) != len(adjacencies):
            raise ValueError("weights must have one entry per adjacency layer")
        if not np.all(np.isfinite(layer_weights)) or np.any(layer_weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if not np.any(layer_weights > 0):
            raise ValueError("at least one layer weight must be positive")

    layers = [validate_adjacency(adjacency) for adjacency in adjacencies]
    shape = layers[0].shape
    if any(layer.shape != shape for layer in layers[1:]):
        raise ValueError("all adjacency layers must have the same shape")

    combined = sparse.csr_matrix(shape, dtype=float)
    for layer, layer_weight in zip(layers, layer_weights, strict=True):
        scale = 1.0
        if layer.data.size and normalize == "max":
            scale = float(layer.data.max())
        elif layer.data.size and normalize == "sum":
            scale = float(layer.data.sum())
        combined = combined + layer * (float(layer_weight) / scale)

    return validate_adjacency(combined, symmetrize=symmetrize)


def knn_indices(X_spatial: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Return directed spatial K-nearest-neighbor indices.

    This compatibility helper returns a dense array. New code that needs
    radius, custom, weighted, or symmetric graphs should use
    :func:`spatial_graph` and :func:`validate_adjacency`.
    """
    graph = spatial_graph(
        X_spatial,
        graph_type="knn",
        n_neighbors=n_neighbors,
        symmetrize="none",
    )
    return np.vstack(adjacency_to_neighbors(graph)).astype(int, copy=False)
