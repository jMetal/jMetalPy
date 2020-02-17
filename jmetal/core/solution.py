from abc import ABC
from typing import List, Generic, TypeVar

from jmetal.util.ckecking import Check

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S], ABC):
    """ Class representing solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return 'Solution(variables={},objectives={},constraints={})'.format(self.variables, self.objectives,
                                                                            self.constraints)


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = BinarySolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total

    def get_binary_string(self) -> str:
        string = ""
        for bit in self.variables[0]:
            string += '1' if bit else '0'
        return string


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, lower_bound: List[float], upper_bound: List[float], number_of_objectives: int,
                 number_of_constraints: int = 0):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, lower_bound: List[int], upper_bound: List[int], number_of_objectives: int,
                 number_of_constraints: int = 0):
        super(IntegerSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class CompositeSolution(Solution):
    """ Class representing solutions composed of a list of solutions. The idea is that each decision  variable can
    be a solution of any type, so we can create mixed solutions (e.g., solutions combining any of the existing
    encodings). The adopted approach has the advantage of easing the reuse of existing variation operators, but all the
    solutions in the list will need to have the same function and constraint violation values.

    It is assumed that problems using instances of this class will properly manage the solutions it contains.
    """

    def __init__(self, solutions: List[Solution]):
        super(CompositeSolution, self).__init__(len(solutions), solutions[0].number_of_objectives,
                                                solutions[0].number_of_constraints)
        Check.is_not_none(solutions)
        Check.collection_is_not_empty(solutions)

        for solution in solutions:
            Check.that(solution.number_of_objectives == solutions[0].number_of_objectives,
                       "The solutions in the list must have the same number of objectives: " + str(
                           solutions[0].number_of_objectives))
            Check.that(solution.number_of_constraints == solutions[0].number_of_constraints,
                       "The solutions in the list must have the same number of constraints: " + str(
                           solutions[0].number_of_constraints))

        self.variables = solutions

    def __copy__(self):
        new_solution = CompositeSolution(self.variables)

        new_solution.objectives = self.objectives[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """ Class representing permutation solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = PermutationSolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
