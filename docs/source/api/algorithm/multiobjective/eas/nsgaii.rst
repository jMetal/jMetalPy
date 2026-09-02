NSGA-II
=======

`NSGA-II <https://doi.org/10.1109/4235.996017>`_ (Non-dominated Sorting Genetic Algorithm II) is
the reference multi-objective evolutionary algorithm: it ranks the population using non-dominated
sorting and uses the crowding distance as a density estimator to preserve diversity.

.. literalinclude:: /../../examples/multiobjective/nsgaii/nsgaii_standard_settings.py
   :language: python

``examples/multiobjective/nsgaii/`` has many more variants: steady-state, dynamic, distributed
(Dask, Spark, multiprocessing), preference-based, real-time plotting, binary and mixed encodings,
constrained problems, and archive-based variants.
