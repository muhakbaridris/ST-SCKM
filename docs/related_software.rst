Related software
================

``stsckm`` occupies a specific design point within centroid clustering,
graph-based learning, spatio-temporal clustering, and regionalization. The
following alternatives answer related but different questions:

* scikit-learn K-means provides unconstrained centroid clustering. Its
  ``AgglomerativeClustering`` estimator can restrict merges with a sparse
  connectivity matrix.
* scikit-learn spectral clustering derives a low-dimensional representation
  from an affinity graph before partitioning. It is preferable when the graph
  itself, rather than explicitly interpretable centroids in the original
  feature space, defines the grouping geometry.
* ``st-dbscan`` implements density-based spatio-temporal clustering for point
  and movement data. It estimates density-connected groups and can return
  noise, so it does not require a fixed number of clusters.
* PySAL ``spopt`` provides regionalization methods including region K-means,
  SKATER, max-p regions, and WardSpatial. These methods are appropriate when
  strict connectivity of areal units is required.
* R package ``ClustGeo`` combines feature and geographic dissimilarities in a
  Ward-like hierarchy. R package ``adespatial`` supplies constrained
  hierarchical clustering from a contiguity edge list.

``GraphRegularizedKMeans`` is intended for numeric feature matrices when the
analyst wants interpretable fixed-``K`` centroids plus a tunable graph
preference. ``STSCKM`` adds an explicit spatial-temporal block interface for
point events. Neither estimator infers noise, guarantees connected clusters,
or replaces a strict regionalization solver. The manuscript reports shared-
input comparisons in wildfire and earthquake examples and discusses the
trade-off between feature separation and graph coherence.
