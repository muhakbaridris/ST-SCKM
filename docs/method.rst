Method
======

For observation ``i`` and candidate cluster ``k``, ST-SCKM combines weighted
squared spatial and temporal distances:

.. math::

   D_{ik} = w_s \|p_i-\mu_k^{(p)}\|^2
          + w_t \|z_i-\mu_k^{(z)}\|^2.

The fitted criterion adds a directed KNN label-disagreement term:

.. math::

   J = \sum_i D_{i,c_i}
     + \lambda \sum_i \sum_{j\in N(i)} I(c_i \ne c_j).

The penalty is soft. It encourages local label agreement but does not guarantee
one connected component per cluster.
