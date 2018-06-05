SMPSO
========================

Common imports:

.. code-block:: python

   from jmetal.core.solution import FloatSolution

   from jmetal.operator.mutation import Polynomial
   from jmetal.component.archive import CrowdingDistanceArchive

SMPSO with standard settings
------------------------------------

.. code-block:: python

   from jmetal.algorithm.multiobjective.smpso import SMPSO

   algorithm = SMPSO(
       problem=problem,
       swarm_size=100,
       max_evaluations=25000,
       mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
       leaders=CrowdingDistanceArchive(100)
   )

   algorithm.run()
   result = algorithm.get_result()

SMPSO/RP with standard settings
------------------------------------

.. code-block:: python

   from jmetal.algorithm.multiobjective.smpso import SMPSORP
   from jmetal.component.archive import CrowdingDistanceArchiveWithReferencePoint

   swarm_size = 100

   reference_points = [[0.8, 0.2], [0.4, 0.6]]
   archives_with_reference_points = []

   for point in reference_points:
       archives_with_reference_points.append(
           CrowdingDistanceArchiveWithReferencePoint(swarm_size, point)
       )

   algorithm = SMPSORP(
       problem=problem,
       swarm_size=swarm_size,
       max_evaluations=25000,
       mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
       reference_points=reference_points,
       leaders=archives_with_reference_points
   )

   algorithm.run()
   result = algorithm.get_result()