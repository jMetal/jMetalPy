import random

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
from jmetal.problem.multiobjective.multiobjective_tsp import MultiObjectiveTSP

"""
.. module:: TSP
   :platform: Unix, Windows
   :synopsis: Single Objective Traveling Salesman problem

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


class TSP(PermutationProblem):
    """Backward-compatible wrapper for single-objective TSP.

    This class delegates to `MultiObjectiveTSP` internally, created with a
    single filename. It preserves the original API (`number_of_objectives()` == 1,
    `evaluate`, `create_solution`) so existing code can switch to it with
    minimal changes.
    """

    def __init__(self, instance: str = None):
        super(TSP, self).__init__()

        if instance is None:
            raise FileNotFoundError("Filename can not be None")

        self._multi = MultiObjectiveTSP([instance])
        # keep compatibility attributes
        self.distance_matrix = self._multi.distance_matrices[0]
        self.number_of_cities = self._multi.number_of_cities
        self.obj_directions = [self.MINIMIZE]

    def number_of_variables(self) -> int:
        return self._multi.number_of_variables()

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        # delegate to the multi-objective implementation (single objective case)
        return self._multi.evaluate(solution)

    def create_solution(self) -> PermutationSolution:
        return self._multi.create_solution()

    def name(self):
        return "Single Objective TSP"
