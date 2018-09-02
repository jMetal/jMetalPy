from abc import ABCMeta
from typing import List, Generic, TypeVar

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S]):
    """ Class representing solutions """

    __metaclass__ = ABCMeta

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables
        self.number_of_constraints = number_of_constraints

        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.attributes = {}

    def __str__(self) -> str:
        solution = 'number_of_objectives: {0} \nnumber_of_variables: {1} \nnumber_of_constraints: {2} \n'.format(
            self.number_of_objectives, self.number_of_variables, self.number_of_constraints
        )
        solution += 'objectives: \n {0} \n'.format(self.objectives)
        solution += 'variables: \n {0}'.format(self.variables)

        return solution


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int=0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = BinarySolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int,
                 lower_bound: List[float], upper_bound: List[float]):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound,
            self.upper_bound)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        return new_solution


class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int,
                 lower_bound: List[int], upper_bound: List[int]):
        super(IntegerSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound,
            self.upper_bound)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        return new_solution
