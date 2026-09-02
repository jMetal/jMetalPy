SMPSO
=====

`SMPSO <https://doi.org/10.1109/MCDM.2009.4938830>`_ (Speed-constrained Multi-objective PSO) is a
particle swarm optimization algorithm that bounds the particle velocity to avoid the swarm
diverging, and uses a crowding-distance-based external archive of leaders.

.. literalinclude:: /../../examples/multiobjective/smpso/smpso_zdt3.py
   :language: python

``examples/multiobjective/smpso/`` also has dynamic, preference-based (SMPSO/RP), and Spark
evaluator variants.
