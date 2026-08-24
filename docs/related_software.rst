Related software
================

``stsckm`` occupies a narrower design point than a general clustering or
regionalization library. The following alternatives answer related but
different questions:

* scikit-learn K-means provides unconstrained centroid clustering. Its
  ``AgglomerativeClustering`` estimator can restrict merges with a sparse
  connectivity matrix.
* ``st-dbscan`` implements density-based spatio-temporal clustering for point
  and movement data. It estimates density-connected groups and can return
  noise, so it does not require a fixed number of clusters.
* PySAL ``spopt`` provides regionalization methods including region K-means,
  SKATER, max-p regions, and WardSpatial. These methods are appropriate when
  strict connectivity of areal units is required.
* R package ``ClustGeo`` combines feature and geographic dissimilarities in a
  Ward-like hierarchy. R package ``adespatial`` supplies constrained
  hierarchical clustering from a contiguity edge list.

``stsckm`` is intended for point observations when the analyst wants a fixed
number of clusters, separate spatial and temporal centroid weights, and a soft
graph penalty that may be tuned rather than an absolute contiguity condition.
It does not infer noise, guarantee connected clusters, or replace a strict
regionalization solver. See the manuscript for citations and an empirical
comparison under shared features and a shared neighborhood graph.
