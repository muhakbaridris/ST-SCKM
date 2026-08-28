"""Parameter search and repeated-fit stability assessment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid

from .diagnostics import graph_diagnostics
from .estimator import STSCKM
from .general import GraphRegularizedKMeans
from .validation import evaluate_labels


@dataclass(frozen=True)
class StabilityResult:
    """Results from fitting one estimator under multiple random seeds."""

    mean_adjusted_rand: float
    pairwise_adjusted_rand: np.ndarray
    seeds: tuple[int, ...]
    labels: tuple[np.ndarray, ...]


def parameter_search(
    X_spatial: np.ndarray,
    X_temporal: np.ndarray,
    param_grid: dict[str, list[object]],
    *,
    estimator: STSCKM | None = None,
    adjacency: np.ndarray | sparse.spmatrix | None = None,
    X_evaluation: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit a deterministic parameter grid and return quality diagnostics."""
    base = STSCKM() if estimator is None else estimator
    evaluation = (
        np.column_stack([X_spatial, X_temporal])
        if X_evaluation is None
        else np.asarray(X_evaluation)
    )
    records: list[dict[str, object]] = []
    for parameters in ParameterGrid(param_grid):
        fitted = clone(base).set_params(**parameters).fit(
            X_spatial,
            X_temporal,
            adjacency=adjacency,
        )
        record: dict[str, object] = dict(parameters)
        record.update(
            {
                "n_iter": fitted.n_iter_,
                "objective": fitted.objective_,
                **evaluate_labels(evaluation, fitted.labels_),
                **graph_diagnostics(fitted.labels_, fitted.adjacency_),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def fit_stability(
    estimator: STSCKM,
    X_spatial: np.ndarray,
    X_temporal: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    adjacency: np.ndarray | sparse.spmatrix | None = None,
) -> StabilityResult:
    """Assess repeated-fit stability using pairwise adjusted Rand indices."""
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    labelings = tuple(
        clone(estimator)
        .set_params(random_state=seed)
        .fit_predict(X_spatial, X_temporal, adjacency=adjacency)
        for seed in seeds
    )
    scores = np.asarray(
        [
            adjusted_rand_score(labelings[i], labelings[j])
            for i, j in combinations(range(len(seeds)), 2)
        ],
        dtype=float,
    )
    return StabilityResult(
        mean_adjusted_rand=float(scores.mean()),
        pairwise_adjusted_rand=scores,
        seeds=tuple(int(seed) for seed in seeds),
        labels=tuple(labels.copy() for labels in labelings),
    )


def graph_parameter_search(
    X: np.ndarray,
    param_grid: dict[str, list[object]],
    *,
    estimator: GraphRegularizedKMeans | None = None,
    adjacency: np.ndarray | sparse.spmatrix | None = None,
    graph_features: np.ndarray | None = None,
    X_evaluation: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit a parameter grid for :class:`GraphRegularizedKMeans`."""
    base = GraphRegularizedKMeans() if estimator is None else estimator
    evaluation = np.asarray(X if X_evaluation is None else X_evaluation)
    records: list[dict[str, object]] = []
    for parameters in ParameterGrid(param_grid):
        fitted = clone(base).set_params(**parameters).fit(
            X,
            adjacency=adjacency,
            graph_features=graph_features,
        )
        record: dict[str, object] = dict(parameters)
        record.update(
            {
                "n_iter": fitted.n_iter_,
                "objective": fitted.objective_,
                **evaluate_labels(evaluation, fitted.labels_),
                **graph_diagnostics(fitted.labels_, fitted.adjacency_),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def graph_fit_stability(
    estimator: GraphRegularizedKMeans,
    X: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    adjacency: np.ndarray | sparse.spmatrix | None = None,
    graph_features: np.ndarray | None = None,
) -> StabilityResult:
    """Assess repeated-fit stability for the general graph estimator."""
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    labelings = tuple(
        clone(estimator)
        .set_params(random_state=seed)
        .fit_predict(X, adjacency=adjacency, graph_features=graph_features)
        for seed in seeds
    )
    scores = np.asarray(
        [
            adjusted_rand_score(labelings[i], labelings[j])
            for i, j in combinations(range(len(seeds)), 2)
        ],
        dtype=float,
    )
    return StabilityResult(
        mean_adjusted_rand=float(scores.mean()),
        pairwise_adjusted_rand=scores,
        seeds=tuple(int(seed) for seed in seeds),
        labels=tuple(labels.copy() for labels in labelings),
    )
