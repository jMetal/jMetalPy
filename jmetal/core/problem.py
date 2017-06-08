import random
from typing import Generic, TypeVar


from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution

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
    """ Class representing binary problems """

    def evaluate(self, solution: BinarySolution) -> None:
        pass

    def create_solution(self) -> BinarySolution:
        pass


class FloatProblem(FloatSolution):
    """ Class representing float problems """
    def __init__(self):
        self.lower_bound = []
        self.upper_bound = []

    def evaluate(self, solution: FloatSolution) -> None:
        pass

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                     self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution


class IntegerProblem(IntegerSolution):
    """ Class representing integer problems """

    def __init__(self):
        self.lower_bound = []
        self.upper_bound = []

    def evaluate(self, solution: IntegerSolution) -> None:
        pass

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                     self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [(int)(random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0)) for i in range(self.number_of_variables)]

        return new_solution