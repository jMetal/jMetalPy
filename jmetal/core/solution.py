from typing import List, Generic, TypeVar

from jmetal.core.problem import FloatProblem, IntegerProblem, Problem, BinaryProblem

__author__ = "Antonio J. Nebro"

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S]):
    """ Class representing solutions """

    def __init__(self, problem: Problem[S]):
        self.number_of_objectives = problem.number_of_objectives
        self.number_of_variables = problem.number_of_variables
        self.number_of_constraints = problem.number_of_constraints
        self.objectives = [0.0 for x in range(self.number_of_objectives)]
        self.variables = [[] for x in range(self.number_of_variables)]
        self.attributes = {}


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, problem: BinaryProblem):
        super(BinarySolution, self).__init__(problem)

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, problem: FloatProblem):
        super(FloatSolution, self).__init__(problem)


class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, int, problem: IntegerProblem):
        super(IntegerSolution, self).__init__(problem)
