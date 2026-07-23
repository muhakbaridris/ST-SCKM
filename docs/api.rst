API reference
=============

Estimator
---------

.. autoclass:: stsckm.STSCKM
   :members: fit, fit_predict, get_objective_history

Distance and graph
------------------

.. autofunction:: stsckm.distance.weighted_spatiotemporal_distance
.. autofunction:: stsckm.graph.knn_indices

Preparation and data
--------------------

.. autofunction:: stsckm.add_default_features
.. autofunction:: stsckm.standardize_features
.. autofunction:: stsckm.generate_sample_wildfire_data
.. autofunction:: stsckm.load_sample_wildfire

Evaluation and profiling
------------------------

.. autofunction:: stsckm.evaluate_labels
.. autofunction:: stsckm.neighbor_disagreement
.. autofunction:: stsckm.assign_risk_labels
