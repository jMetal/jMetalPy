from typing import List, Generic, TypeVar

__author__ = "Antonio J. Nebro"

BitSet = List[bool]
T = TypeVar('T')


class Solution(Generic[T]):
    """ Class representing solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables
        self.objectives = [0.0 for x in range(self.number_of_objectives)]
        self.variables = [[] for x in range(self.number_of_variables)]
        self.attributes = {}


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives)

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, lower_bound: List=list(),
                 upper_bound: List=list()):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class IntegerSolution(Solution[int]):
    def __init__(self, number_of_variables:int, number_of_objectives: int, lower_bound: List=list(),
                 upper_bound: List=list()):
        super(IntegerSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound