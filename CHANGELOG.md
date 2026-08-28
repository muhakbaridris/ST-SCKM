# Changelog

## 2.1.0 - 2026-08-29

- Added `GraphRegularizedKMeans`, a domain-general estimator that separates
  centroid features from graph-construction features and supports custom sparse
  adjacency matrices.
- Added feature-level weights, centroid-only out-of-sample assignment, and
  scikit-learn-compatible fitted metadata for the general estimator.
- Added `combine_adjacencies` for weighted multi-layer graphs with explicit
  per-layer normalization.
- Refactored `STSCKM` and the general estimator to share one tested
  optimization engine while retaining the established STSCKM interface.
- Added general parameter-search and repeated-fit stability helpers.
- Added a reproducible non-wildfire illustration using an archived 924-event
  USGS California earthquake catalog with source query and SHA-256 provenance.
- Expanded the cross-method comparison, documentation, tests, replication
  materials, and manuscript around multi-view and layered-graph use cases.
- Added continuous integration across Python 3.10 through 3.13, with separate
  lint, documentation, build, and distribution checks.

## 2.0.0 - 2026-08-25

- Generalized the neighborhood layer to sparse KNN, radius, and caller-supplied
  weighted adjacency graphs with explicit directed, union, and mutual modes.
- Added graph disagreement, cluster connectivity, fragmentation, parameter
  search, and repeated-fit stability diagnostics.
- Added synchronous and sequential update schemes and an unregularized
  centroid-distance transform.
- Defined graph regularization on symmetric edge weights and recorded the
  final-centroid objective after each pass.
- Rebuilt documentation and replication materials around one executable entry
  point, exact manuscript listings, an independently runnable example, pinned
  requirements, archived checks, and session information.
- Added empirical comparison with unconstrained K-means and
  connectivity-constrained Ward clustering.
- Added an end-to-end worked analysis, pairwise partition-agreement results,
  row-order sensitivity checks, and a reproducible scaling benchmark.

## 1.0.1 - 2026-07-24

- Updated installation documentation after publication on PyPI.
- No statistical or API behavior changed from version 1.0.0.

## 1.0.0 - 2026-07-24

- Added an installable `stsckm` package using a standard `src` layout.
- Added a scikit-learn-compatible estimator interface.
- Added explicit validation for cluster counts, weights, penalties, neighbors,
  stopping controls, and aligned inputs.
- Made empty-cluster replacement use the estimator random state and one
  observation for both spatial and temporal centroid components.
- Added distance, KNN graph, preprocessing, evaluation, profiling, dataset,
  and optional plotting helpers.
- Added unit tests, documentation, continuous integration, and a single
  replication entry point.
