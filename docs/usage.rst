Usage
=====

The domain-general estimator accepts one centroid-feature matrix and can build
its graph from a different representation:

.. code-block:: python

   from stsckm import GraphRegularizedKMeans

   model = GraphRegularizedKMeans(
       n_clusters=4,
       graph_penalty=1.0,
       feature_weights=[1.0, 1.0, 0.5],
       n_neighbors=6,
       graph_symmetrize="union",
       random_state=42,
   ).fit(X, graph_features=graph_features)

``predict(X_new)`` uses weighted centroid distances only. It does not invent a
graph connecting new observations to the fitted sample.

The spatio-temporal specialization accepts spatial and temporal matrices
separately:

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

Layered graph
-------------

.. code-block:: python

   from stsckm import combine_adjacencies

   adjacency = combine_adjacencies(
       [spatial_layer, temporal_layer],
       weights=[0.8, 0.2],
       normalize="max",
       symmetrize="union",
   )

Layer weights and normalization determine the relative contribution of each
relationship type. Store or export the returned adjacency matrix when an exact
analysis must be reproduced.

Cross-domain illustration
-------------------------

``python examples/earthquake_catalog.py`` runs a second complete example on an
archived USGS earthquake catalog. The example uses magnitude, depth, and event
time as centroid features and combines geographic and temporal graph layers.
It is a software illustration, not a seismic-hazard model.

Spatial and temporal matrices should be scaled deliberately. Geographic
coordinates should be projected with a coordinate reference system suitable
for the study region.
