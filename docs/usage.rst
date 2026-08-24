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

Inspecting fitted state
-----------------------

``labels_`` contains the final partition. Spatial and temporal centroids are
stored separately in ``cluster_centers_spatial_`` and
``cluster_centers_temporal_``. ``adjacency_`` is the validated input graph,
whereas ``regularization_adjacency_`` is the symmetric matrix used by the
penalty. ``objective_history_`` and ``graph_diagnostics_`` support convergence
and fragmentation checks.

``transform(X_spatial, X_temporal)`` returns weighted centroid costs. It does
not include a graph penalty and is therefore an inspection tool rather than an
implicit graph-regularized prediction rule.

Custom weighted graph
---------------------

.. code-block:: python

   import numpy as np
   from stsckm import STSCKM

   adjacency = np.zeros((len(X_spatial), len(X_spatial)))
   adjacency[0, 1] = 2.0
   adjacency[1, 0] = 1.0

   model = STSCKM(
       n_clusters=4,
       lambda_spatial=0.75,
       graph_symmetrize="none",
       random_state=42,
   ).fit(X_spatial, X_temporal, adjacency=adjacency)

With ``graph_symmetrize="none"``, the original asymmetric matrix remains in
``adjacency_`` and its average with the transpose is stored in
``regularization_adjacency_``. Multiplying all edge weights by a positive
constant has the same objective-scale effect as multiplying
``lambda_spatial`` by that constant.

Spatial and temporal matrices should be scaled deliberately. Geographic
coordinates should be projected with a coordinate reference system suitable
for the study region.
