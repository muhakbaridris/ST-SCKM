"""Graph-regularized spatio-temporal clustering in Python."""

from ._version import __version__
from .datasets import load_sample_wildfire
from .diagnostics import adjacency_disagreement, cluster_connectivity, graph_diagnostics
from .estimator import STSCKM
from .graph import adjacency_to_neighbors, knn_indices, spatial_graph, validate_adjacency
from .model_selection import StabilityResult, fit_stability, parameter_search
from .preprocessing import (
    add_default_features,
    generate_sample_wildfire_data,
    standardize_features,
)
from .validation import (
    RISK_LABELS,
    assign_risk_labels,
    evaluate_labels,
    neighbor_disagreement,
)

__all__ = [
    "RISK_LABELS",
    "STSCKM",
    "StabilityResult",
    "__version__",
    "adjacency_disagreement",
    "adjacency_to_neighbors",
    "add_default_features",
    "assign_risk_labels",
    "cluster_connectivity",
    "evaluate_labels",
    "fit_stability",
    "generate_sample_wildfire_data",
    "graph_diagnostics",
    "knn_indices",
    "load_sample_wildfire",
    "neighbor_disagreement",
    "parameter_search",
    "spatial_graph",
    "standardize_features",
    "validate_adjacency",
]
