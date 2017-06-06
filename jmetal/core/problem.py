import random
from typing import Generic, TypeVar

from jmetal.core.solution import BinarySolution, FloatSolution

__author__ = "Antonio J. Nebro"

S = TypeVar('S')

class Problem(Generic[S]):
    """ Class representing problems """

    def __init__(self):
        self.number_of_variables = 0
        self.number_of_objectives = 0
        self.number_of_constraints = 0

    def evaluate(self, solution: S) -> None:
        pass

    def create_solution(self) -> S:
        pass

class BinaryProblem(BinarySolution):
    """ Class representing float problems """

    def evaluate(self, solution: BinarySolution) -> None:
        pass

    def create_solution(self) -> BinarySolution:
        pass


class FloatProblem(FloatSolution):
    """ Class representing float problems """

    lower_bound = []
    upper_bound = []

    def __init__(self):
        pass

    def evaluate(self, solution: FloatSolution) -> None:
        pass

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution
