Reproducibility
===============

The submitted replication archive has its own pinned ``requirements.txt``.
After installing the submitted source archive, change into ``replication`` and
run the single entry point:

.. code-block:: console

   $ cd replication
   $ python run_all.py

It executes every manuscript listing, regenerates the sensitivity, software
comparison, graph-interface, and stability tables, recreates every data-driven
figure, records the active environment, and checks all numerical tables against
the archived expected outputs. No network access is required.
