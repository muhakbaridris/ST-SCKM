# stsckm

[![Continuous integration](https://github.com/muhakbaridris/ST-SCKM/actions/workflows/ci.yml/badge.svg)](https://github.com/muhakbaridris/ST-SCKM/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/stsckm.svg)](https://pypi.org/project/stsckm/)
[![Python versions](https://img.shields.io/pypi/pyversions/stsckm.svg)](https://pypi.org/project/stsckm/)

`stsckm` is a Python framework for graph-regularized K-means clustering. It
supports general feature matrices, multi-view representations, layered sparse
graphs, and a dedicated spatio-temporal estimator for point-event data.

`GraphRegularizedKMeans` separates the variables used to define centroids from
the representation used to construct the neighborhood graph. `STSCKM` remains
the backward-compatible specialization that applies separate spatial and
temporal distance weights. Both estimators use one shared optimization engine
and a soft label-disagreement penalty.

The method was introduced in:

> Idris, M. A., Aidi, M. N., and Djuraidah, A. (2026). Performance Evaluation
> of ST-DBSCAN and Spatio-Temporal Spatially Constrained K-Means (ST-SCKM) for
> Wildfire Risk Zoning and Resilience Analysis. *Journal of Safety Science and
> Resilience*. <https://doi.org/10.1016/j.jnlssr.2026.100357>

The package generalizes the reference implementation into reusable graph,
diagnostic, selection, and multi-view components. Version 2.1.0 includes
reproducible wildfire and earthquake illustrations; the estimators themselves
are not tied to either domain.

## What the package provides

- a scikit-learn-style `GraphRegularizedKMeans` estimator for arbitrary numeric
  feature matrices;
- a backward-compatible `STSCKM` estimator for aligned spatial and temporal
  matrices;
- separate centroid features and graph-construction features;
- directed, union, or mutual K-nearest-neighbor graphs and radius graphs;
- caller-supplied dense or SciPy sparse weighted adjacency matrices;
- weighted multi-layer graph composition with optional per-layer
  normalization;
- sequential or synchronous label-update schemes;
- graph disagreement, connected-component, and fragmentation diagnostics;
- deterministic parameter-grid summaries and repeated-fit stability checks;
- conventional silhouette, Calinski-Harabasz, and Davies-Bouldin indices;
- explicit separation between clustering variables and post hoc profile
  variables; and
- a synthetic wildfire dataset, an archived USGS earthquake catalog, tests,
  documentation, examples, and a complete replication archive.

The graph penalty is soft. It encourages neighboring observations to share a
label but does **not** guarantee that every cluster is connected. Use a strict
regionalization method when connected regions are mandatory.

## Installation

Install the published package from PyPI:

```bash
python -m pip install stsckm
```

Optional plotting support is available with:

```bash
python -m pip install "stsckm[plot]"
```

For an editable development installation from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows, use `.venv\Scripts\activate`.

## General graph-regularized estimator

```python
import numpy as np

from stsckm import GraphRegularizedKMeans

# X defines centroid fit. graph_features can be a different representation.
X = np.array(
    [
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.9],
        [0.2, 0.1, 1.1],
        [4.0, 4.1, 8.0],
        [4.1, 4.0, 8.2],
        [4.2, 4.1, 7.9],
    ]
)
graph_features = X[:, :2]
model = GraphRegularizedKMeans(
    n_clusters=2,
    graph_penalty=1.0,
    feature_weights=[1.0, 1.0, 0.5],
    n_neighbors=2,
    graph_symmetrize="union",
    random_state=42,
).fit(X, graph_features=graph_features)

print(model.graph_diagnostics_)
```

The general estimator implements `fit`, `fit_predict`, `transform`, and
`predict`. Its `predict` method uses centroid distances only because no graph
for new observations is implied by the fitted sample.

## Spatio-temporal specialization

```python
import numpy as np

from stsckm import (
    STSCKM,
    add_default_features,
    evaluate_labels,
    load_sample_wildfire,
    standardize_features,
)

events = add_default_features(load_sample_wildfire())
X_spatial, spatial_scaler = standardize_features(
    events, ["x_proj", "y_proj"]
)
X_temporal, temporal_scaler = standardize_features(events, ["time_days"])

model = STSCKM(
    n_clusters=4,
    spatial_weight=0.5,
    temporal_weight=1.5,
    lambda_spatial=1.0,
    graph_type="knn",
    n_neighbors=5,
    graph_symmetrize="union",
    random_state=42,
).fit(X_spatial, X_temporal)

metrics = evaluate_labels(
    np.column_stack([X_spatial, X_temporal]), model.labels_
)
print(metrics)
print(model.graph_diagnostics_)
```

The fitted object stores `labels_`, separate spatial and temporal centroids,
the original sparse `adjacency_`, the symmetric
`regularization_adjacency_`, an objective history, and graph diagnostics.
`transform()` returns unregularized weighted distances to the fitted
centroids. The package deliberately does not define out-of-sample `predict()`
because prediction requires an explicit rule for connecting new observations
to the fitted graph.

## Radius, custom, and layered graphs

```python
from stsckm import STSCKM, spatial_graph

radius_model = STSCKM(
    n_clusters=4,
    graph_type="radius",
    radius=0.20,
    lambda_spatial=1.0,
    random_state=42,
).fit(X_spatial, X_temporal)

adjacency = spatial_graph(
    X_spatial,
    graph_type="knn",
    n_neighbors=8,
    symmetrize="mutual",
)
custom_model = STSCKM(n_clusters=4, random_state=42).fit(
    X_spatial,
    X_temporal,
    adjacency=adjacency,
)
```

Multiple relationship layers can be combined before fitting:

```python
from stsckm import combine_adjacencies

layered_graph = combine_adjacencies(
    [spatial_layer, temporal_layer],
    weights=[0.8, 0.2],
    normalize="max",
    symmetrize="union",
)
```

Non-negative edge weights in a custom graph enter the disagreement penalty.
Diagonal entries are removed during validation. A complete asymmetric
six-observation example is available in
[`examples/custom_weighted_graph.py`](examples/custom_weighted_graph.py).

## Non-wildfire example

`examples/earthquake_catalog.py` applies `GraphRegularizedKMeans` to an
archived 924-event USGS California earthquake catalog. Centroid fit uses
magnitude, depth, and event time, while a layered graph combines geographic and
temporal neighborhoods. The query, retrieval date, and SHA-256 digest are
recorded in [`data/README.md`](data/README.md). This example demonstrates the
software interface and is not a seismic-hazard analysis.

```bash
python examples/earthquake_catalog.py
```

## Parameter selection and stability

```python
from stsckm import STSCKM, fit_stability, parameter_search

search = parameter_search(
    X_spatial,
    X_temporal,
    {
        "n_clusters": [3, 4],
        "lambda_spatial": [0.0, 0.5, 1.0, 2.0],
        "graph_symmetrize": ["union"],
    },
    estimator=STSCKM(n_neighbors=5, random_state=42),
)

stability = fit_stability(
    STSCKM(n_clusters=4, graph_symmetrize="union"),
    X_spatial,
    X_temporal,
    seeds=(0, 1, 2, 3, 4),
)
print(search)
print(stability.mean_adjusted_rand)
```

Internal separation and graph coherence quantify different properties. A
parameter choice should therefore consider both, together with domain checks
when available.

## Related software

The package does not replace all spatial clustering tools:

| Software or method | Main mechanism | Prefer it when |
|---|---|---|
| scikit-learn K-means | Unconstrained centroid clustering | No graph preference is required |
| scikit-learn connectivity-constrained Ward | Only connected clusters may merge | A hierarchical partition under a connectivity graph is appropriate |
| `st-dbscan` | Spatial and temporal density thresholds | Noise detection and an unknown cluster count are central |
| PySAL `spopt` | Strict regionalization algorithms | Areal units must form connected regions |
| R `ClustGeo` | Feature and geographic dissimilarities | A Ward-like hierarchy with soft geographic dissimilarity is preferred |
| R `adespatial::constr.hclust` | Contiguity-constrained agglomeration | Spatial or chronological contiguity must restrict each merge |
| `stsckm` | Soft graph disagreement plus general or space-time centroids | Fixed-`K` feature or event data need tunable graph coherence and explicit graph inspection |

The JSS manuscript gives full citations, a feature comparison, limitations,
and shared-input empirical illustrations in two application domains.

## Reproducing the manuscript

The review archive is self-contained. After installing its submitted source
distribution, run all commands from the `replication` folder:

```bash
cd replication
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` executes the manuscript listings, runs the independently callable
`examples/run_example.py`, recreates all numerical tables and data-driven
figures, records dependency versions, and checks platform-invariant results
against the archived CSVs in `expected/`. It also regenerates a small scaling
benchmark and the earthquake comparison without comparing wall times for exact
equality. See
[`replication/README.md`](replication/README.md) for the exact reviewer
workflow and output map.

For a single end-to-end script that prepares the bundled data, fits the model,
writes diagnostics, screens a transparent parameter grid, and saves auditable
outputs, run `python replication/worked_analysis.py` from the `replication`
folder.

## Tests, documentation, and build

```bash
python -m pytest
python -m ruff check src tests replication
python -m sphinx -W -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

## License and author

`stsckm` is released under the MIT license, which is GPL-compatible.

Muh Akbar Idris, IPB University<br>
ORCID: <https://orcid.org/0009-0000-2995-1975>
