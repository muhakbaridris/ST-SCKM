# Changelog

## 2.0.0 - Unreleased

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
