# stsckm

`stsckm` is a Python implementation of spatio-temporal spatially constrained
K-means (ST-SCKM) for point-event data. It combines weighted spatial and
temporal squared distances with a soft label-disagreement penalty on a spatial
K-nearest-neighbor graph.

The package implements the method associated with:

> Idris, M. A., Aidi, M. N., and Djuraidah, A. (2026). Performance Evaluation
> of ST-DBSCAN and Spatio-Temporal Spatially Constrained K-Means (ST-SCKM) for
> Wildfire Risk Zoning and Resilience Analysis. *Journal of Safety Science and
> Resilience*. <https://doi.org/10.1016/j.jnlssr.2026.100357>

## Scope

The estimator is intended for aligned spatial and temporal numeric matrices.
The KNN penalty encourages neighboring observations to share labels but does
not guarantee that every cluster is a single connected component. Context
variables such as event intensity can be retained for post hoc profiling
without entering the clustering objective.

## Installation

Install the source checkout:

```bash
python -m pip install .
```

For development, testing, plotting, and documentation:

```bash
python -m pip install -e ".[dev]"
```

The package has not yet been released on PyPI. Do not use
`pip install stsckm` until a release is published there.

## Quick start

```python
import numpy as np

from stsckm import (
    STSCKM,
    add_default_features,
    evaluate_labels,
    load_sample_wildfire,
    standardize_features,
)

frame = add_default_features(load_sample_wildfire())
X_spatial, spatial_scaler = standardize_features(
    frame, ["x_proj", "y_proj"]
)
X_temporal, temporal_scaler = standardize_features(
    frame, ["time_days"]
)

model = STSCKM(
    n_clusters=4,
    spatial_weight=0.5,
    temporal_weight=1.5,
    lambda_spatial=1.0,
    n_neighbors=5,
    random_state=42,
)
labels = model.fit_predict(X_spatial, X_temporal)

metrics = evaluate_labels(
    np.column_stack([X_spatial, X_temporal]),
    labels,
)
print(model.n_iter_)
print(metrics)
```

## Main API

- `STSCKM.fit(X_spatial, X_temporal)`
- `STSCKM.fit_predict(X_spatial, X_temporal)`
- `weighted_spatiotemporal_distance(...)`
- `knn_indices(...)`
- `evaluate_labels(...)`
- `neighbor_disagreement(...)`
- `assign_risk_labels(...)`
- `load_sample_wildfire()`

The estimator follows scikit-learn parameter conventions and inherits from
`BaseEstimator` and `ClusterMixin`. Version 1.0.0 intentionally does not expose
out-of-sample `predict()`: defining neighbors for new points requires a
separate modeling choice.

## Reproduction

Run all manuscript-scale numerical results and figures with one command:

```bash
python replication/run_all.py
```

The script writes its outputs to `replication/output/` and checks the
sensitivity table against `replication/expected/sensitivity.csv`.

## Tests and build

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## License

MIT. This is a GPL-compatible free-software license.

## Author

Muh Akbar Idris<br>
IPB University<br>
ORCID: <https://orcid.org/0009-0000-2995-1975>
