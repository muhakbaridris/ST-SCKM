Usage
=====

The main estimator accepts spatial and temporal matrices separately:

.. code-block:: python

   from stsckm import STSCKM, graph_diagnostics

   model = STSCKM(
       n_clusters=4,
       spatial_weight=0.5,
       temporal_weight=1.5,
       lambda_spatial=1.0,
       n_neighbors=5,
       graph_symmetrize="union",
       random_state=42,
   ).fit(X_spatial, X_temporal)

   print(model.graph_diagnostics_)

Use ``graph_type="radius"`` with a positive ``radius`` for a radius graph.
A custom dense or sparse graph can be passed as
``model.fit(X_spatial, X_temporal, adjacency=adjacency)``. Edge weights enter
the disagreement penalty.

``parameter_search()`` reports conventional internal indices together with
graph coherence and fragmentation. ``fit_stability()`` compares repeated
fits with the adjusted Rand index.

Spatial and temporal matrices should be scaled deliberately. Geographic
coordinates should be projected with a coordinate reference system suitable
for the study region.
