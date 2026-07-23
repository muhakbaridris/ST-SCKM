# Changelog

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
