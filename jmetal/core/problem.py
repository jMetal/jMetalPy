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

    def evaluate(self, solution: FloatSolution) -> None:
        pass

    def create_solution(self) -> FloatSolution:
        pass

