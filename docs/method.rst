Method
======

For observation ``i`` and candidate cluster ``k``, ST-SCKM combines weighted
squared spatial and temporal distances:

.. math::

   D_{ik} = w_s \|p_i-\mu_k^{(p)}\|^2
          + w_t \|z_i-\mu_k^{(z)}\|^2.

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
