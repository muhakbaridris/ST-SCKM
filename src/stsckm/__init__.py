"""Graph-regularized K-means for structured and spatio-temporal data."""

from ._version import __version__
from .datasets import (
    EARTHQUAKE_DATA_SHA256,
    EARTHQUAKE_DATA_SOURCE,
    load_sample_earthquakes,
    load_sample_wildfire,
)
from .diagnostics import adjacency_disagreement, cluster_connectivity, graph_diagnostics
from .estimator import STSCKM
from .general import GraphRegularizedKMeans
from .graph import (
    adjacency_to_neighbors,
    combine_adjacencies,
    knn_indices,
    spatial_graph,
    validate_adjacency,
)
from .model_selection import (
    StabilityResult,
    fit_stability,
    graph_fit_stability,
    graph_parameter_search,
    parameter_search,
)
from .preprocessing import (
    add_default_features,
    add_point_event_features,
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
    "EARTHQUAKE_DATA_SHA256",
    "EARTHQUAKE_DATA_SOURCE",
    "GraphRegularizedKMeans",
    "STSCKM",
    "StabilityResult",
    "__version__",
    "adjacency_disagreement",
    "adjacency_to_neighbors",
    "add_default_features",
    "add_point_event_features",
    "assign_risk_labels",
    "cluster_connectivity",
    "combine_adjacencies",
    "evaluate_labels",
    "fit_stability",
    "generate_sample_wildfire_data",
    "graph_diagnostics",
    "graph_fit_stability",
    "graph_parameter_search",
    "knn_indices",
    "load_sample_earthquakes",
    "load_sample_wildfire",
    "neighbor_disagreement",
    "parameter_search",
    "spatial_graph",
    "standardize_features",
    "validate_adjacency",
]
