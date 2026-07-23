"""Spatially constrained spatio-temporal K-means clustering."""

from ._version import __version__
from .datasets import load_sample_wildfire
from .estimator import STSCKM
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
    "__version__",
    "add_default_features",
    "assign_risk_labels",
    "evaluate_labels",
    "generate_sample_wildfire_data",
    "load_sample_wildfire",
    "neighbor_disagreement",
    "standardize_features",
]
