API reference
=============

Estimators
----------

.. autoclass:: stsckm.GraphRegularizedKMeans
   :members: fit, fit_predict, transform, predict, get_objective_history

.. autoclass:: stsckm.STSCKM
   :members: fit, fit_predict, transform, get_objective_history

Distance and graph
------------------

.. autofunction:: stsckm.distance.weighted_spatiotemporal_distance
.. autofunction:: stsckm.graph.knn_indices
.. autofunction:: stsckm.spatial_graph
.. autofunction:: stsckm.validate_adjacency
.. autofunction:: stsckm.adjacency_to_neighbors
.. autofunction:: stsckm.combine_adjacencies

Preparation and data
--------------------

.. autofunction:: stsckm.add_default_features
.. autofunction:: stsckm.add_point_event_features
.. autofunction:: stsckm.standardize_features
.. autofunction:: stsckm.generate_sample_wildfire_data
.. autofunction:: stsckm.load_sample_wildfire
.. autofunction:: stsckm.load_sample_earthquakes

Evaluation and profiling
------------------------

.. autofunction:: stsckm.evaluate_labels
.. autofunction:: stsckm.neighbor_disagreement
.. autofunction:: stsckm.adjacency_disagreement
.. autofunction:: stsckm.cluster_connectivity
.. autofunction:: stsckm.graph_diagnostics
.. autofunction:: stsckm.assign_risk_labels

Selection and stability
-----------------------

.. autofunction:: stsckm.parameter_search
.. autofunction:: stsckm.fit_stability
.. autofunction:: stsckm.graph_parameter_search
.. autofunction:: stsckm.graph_fit_stability
