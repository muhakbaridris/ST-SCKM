Method
======

For observation ``i`` and candidate cluster ``k``, the general estimator uses
feature-level weighted squared distance:

.. math::

   D_{ik} = \sum_{r=1}^{q} v_r (x_{ir}-\mu_{kr})^2.

Let ``A`` be a non-negative adjacency matrix and define the symmetric
regularization weights ``W = (A + A.T) / 2``. The fitted criterion adds a
weighted graph disagreement term:

.. math::

   J = \sum_i D_{i,c_i}
     + \lambda \sum_{i<j} W_{ij} I(c_i \ne c_j).

The penalty is soft. It encourages local label agreement but does not guarantee
one connected component per cluster. ``stsckm`` can construct directed,
union, or mutual KNN graphs and radius graphs, or accept a caller-supplied
dense or SciPy sparse adjacency matrix. Directional input is converted to the
symmetric weights above because label disagreement itself has no direction.

``STSCKM`` is a specialization with two feature blocks. Repeating
``spatial_weight`` for every spatial column and ``temporal_weight`` for every
temporal column gives

.. math::

   D_{ik} = w_s \|p_i-\mu_k^{(p)}\|^2
          + w_t \|z_i-\mu_k^{(z)}\|^2.

Both public estimators call the same optimization engine. Multiple adjacency
layers can be normalized and combined before fitting. Because graph scaling
and the penalty multiplier are not separately identifiable, every analysis
should report both the layer normalization and the final graph penalty.
