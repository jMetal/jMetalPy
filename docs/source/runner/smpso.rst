SMPSO
========================

Common imports:

.. code-block:: python

   from jmetal.operator import Polynomial

   from jmetal.problem import ZDT1

SMPSO with standard settings
------------------------------------

.. code-block:: python

   from jmetal.algorithm import SMPSO
   from jmetal.component import CrowdingDistanceArchive

   algorithm = SMPSO(
       problem=ZDT1(rf_path='resources/reference_front/ZDT1.pf'),
       swarm_size=100,
       max_evaluations=25000,
       mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
       leaders=CrowdingDistanceArchive(100)
   )

   algorithm.run()
   front = algorithm.get_result()

SMPSO/RP with standard settings
------------------------------------

.. code-block:: python

   from jmetal.algorithm import SMPSORP
   from jmetal.component import CrowdingDistanceArchiveWithReferencePoint

   swarm_size = 100

   reference_points = [[0.8, 0.2], [0.4, 0.6]]
   archives_with_reference_points = []

   for point in reference_points:
       archives_with_reference_points.append(
           CrowdingDistanceArchiveWithReferencePoint(swarm_size, point)
       )

   algorithm = SMPSORP(
       problem=ZDT1(rf_path='resources/reference_front/ZDT1.pf'),
       swarm_size=swarm_size,
       max_evaluations=25000,
       mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
       reference_points=reference_points,
       leaders=archives_with_reference_points
   )

   algorithm.run()
   front = algorithm.get_result()