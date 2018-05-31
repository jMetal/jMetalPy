NSGA-II
========================

Common imports for these examples:

.. code-block:: python

   from jmetal.algorithm.multiobjective.nsgaii import NSGAII
   from jmetal.core.solution import FloatSolution

   from jmetal.operator.mutation import Polynomial
   from jmetal.operator.crossover import SBX
   from jmetal.operator.selection import BinaryTournamentSelection
   from jmetal.util.comparator import RankingAndCrowdingDistanceComparator

NSGA-II with standard settings
------------------------------------

.. code-block:: python

   algorithm = NSGAII[FloatSolution, List[FloatSolution]](
      problem=problem,
      population_size=100,
      max_evaluations=25000,
      mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
      crossover=SBX(1.0, distribution_index=20),
      selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())
   )

   algorithm.run()
   result = algorithm.get_result()

NSGA-II stopping by time
------------------------------------

.. code-block:: python

   from typing import List, TypeVar

   S = TypeVar('S')
   R = TypeVar(List[S])

   def main():
      class NSGA2b(NSGAII[S, R]):
         def is_stopping_condition_reached(self):
            # Re-define the stopping condition
            reached = [False, True][self.get_current_computing_time() > 4]

            if reached:
               logger.info("Stopping condition reached!")

            return reached

      algorithm = NSGA2b[FloatSolution, List[FloatSolution]](
         problem,
         population_size=100,
         max_evaluations=25000,
         mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
         crossover=SBX(1.0, distribution_index=20),
         selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())
      )

      algorithm.run()
      result = algorithm.get_result()
