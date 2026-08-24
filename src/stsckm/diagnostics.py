"""Graph-aware diagnostics for spatially coherent partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .graph import validate_adjacency


def adjacency_disagreement(
    labels: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
) -> float:
    """Return the weighted fraction of graph edges joining unlike labels."""
    labels = _validate_labels(labels)
    graph = validate_adjacency(adjacency, n_samples=len(labels))
    total = float(graph.data.sum())
    if total == 0:
        return float("nan")
    rows, columns = graph.nonzero()
    return float(graph.data[labels[rows] != labels[columns]].sum() / total)


def cluster_connectivity(
    labels: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
) -> pd.DataFrame:
    """Summarize connected components induced by each non-noise cluster."""
    labels = _validate_labels(labels)
    graph = validate_adjacency(adjacency, n_samples=len(labels), symmetrize="union")
    records: list[dict[str, float | int]] = []
    for cluster in np.unique(labels[labels != -1]):
        members = np.flatnonzero(labels == cluster)
        subgraph = graph[members][:, members]
        n_components, membership = connected_components(
            subgraph,
            directed=False,
            return_labels=True,
        )
        counts = np.bincount(membership)
        largest = int(counts.max())
        records.append(
            {
                "cluster": int(cluster),
                "size": int(len(members)),
                "n_components": int(n_components),
                "largest_component_size": largest,
                "largest_component_fraction": float(largest / len(members)),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "cluster",
            "size",
            "n_components",
            "largest_component_size",
            "largest_component_fraction",
        ],
    )


def graph_diagnostics(
    labels: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
) -> dict[str, float | int]:
    """Return scalar graph coherence and fragmentation diagnostics."""
    labels = _validate_labels(labels)
    connectivity = cluster_connectivity(labels, adjacency)
    disagreement = adjacency_disagreement(labels, adjacency)
    if connectivity.empty:
        weighted_largest = float("nan")
        total_components = 0
        disconnected = 0
    else:
        weighted_largest = float(
            connectivity["largest_component_size"].sum() / connectivity["size"].sum()
        )
        total_components = int(connectivity["n_components"].sum())
        disconnected = int((connectivity["n_components"] > 1).sum())
    return {
        "edge_disagreement": disagreement,
        "neighbor_agreement": (
            1.0 - disagreement if np.isfinite(disagreement) else float("nan")
        ),
        "n_components_total": total_components,
        "disconnected_clusters": disconnected,
        "largest_component_fraction": weighted_largest,
    }


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if values.size == 0:
        raise ValueError("labels cannot be empty")
    return values
