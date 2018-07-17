NSGA-II
========================

Common imports for these examples:

.. code-block:: python

   from jmetal.algorithm import NSGAII
   from jmetal.operator import Polynomial, SBX, BinaryTournamentSelection
   from jmetal.component import RankingAndCrowdingDistanceComparator

   from jmetal.problem import ZDT1

NSGA-II with standard settings
------------------------------------

.. code-block:: python

   algorithm = NSGAII(
      problem=ZDT1(rf_path='resources/reference_front/ZDT1.pf'),
      population_size=100,
      max_evaluations=25000,
      mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
      crossover=SBX(probability=1.0, distribution_index=20),
      selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
   )

   algorithm.run()
   front = algorithm.get_result()

NSGA-II stopping by time
------------------------------------

.. code-block:: python

   class NSGA2b(NSGAII):
      def is_stopping_condition_reached(self):
         # Re-define the stopping condition
         return [False, True][self.get_current_computing_time() > 4]

   algorithm = NSGA2b(
      problem=ZDT1(rf_path='resources/reference_front/ZDT1.pf'),
      population_size=100,
      max_evaluations=25000,
      mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
      crossover=SBX(probability=1.0, distribution_index=20),
      selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
   )

   algorithm.run()
   front = algorithm.get_result()
