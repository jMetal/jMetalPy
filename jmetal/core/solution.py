from abc import ABC
from typing import Generic, List, TypeVar

from jmetal.util.ckecking import Check

BitSet = List[bool]
S = TypeVar("S")


class Solution(Generic[S], ABC):
    """Class representing solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.variables = [[] for _ in range(number_of_variables)]
        self.objectives = [0.0 for _ in range(number_of_objectives)]
        self.constraints = [0.0 for _ in range(number_of_constraints)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return "Solution(variables={},objectives={},constraints={})".format(
            self.variables, self.objectives, self.constraints
        )


class BinarySolution(Solution[BitSet]):
    """Class representing float solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

        self.bits_per_variable = []

    def __copy__(self):
        new_solution = BinarySolution(len(self.variables), len(self.objectives), len(self.constraints))
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()
        new_solution.bits_per_variable = self.bits_per_variable

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total

    def get_binary_string(self) -> str:
        string = ""
        for bit in self.variables[0]:
            string += "1" if bit else "0"
        return string

    def cardinality(self, variable_index) -> int:
        return sum(1 for _ in self.variables[variable_index] if _)


class FloatSolution(Solution[float]):
    """Class representing float solutions"""

    def __init__(
            self,
            lower_bound: List[float],
            upper_bound: List[float],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, len(self.objectives), len(self.constraints)
        )
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.lower_bound = self.lower_bound
        new_solution.upper_bound = self.upper_bound

        new_solution.attributes = self.attributes.copy()

        return new_solution


class IntegerSolution(Solution[int]):
    """Class representing integer solutions"""

    def __init__(
            self,
            lower_bound: List[int],
            upper_bound: List[int],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ):
        super(IntegerSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, len(self.objectives), len(self.constraints)
        )
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.lower_bound = self.lower_bound
        new_solution.upper_bound = self.upper_bound

        new_solution.attributes = self.attributes.copy()

        return new_solution


class CompositeSolution(Solution):
    """Class representing solutions composed of a list of solutions. The idea is that each decision  variable can
    be a solution of any type, so we can create mixed solutions (e.g., solutions combining any of the existing
    encodings). The adopted approach has the advantage of easing the reuse of existing variation operators, but all the
    solutions in the list will need to have the same function and constraint violation values.

    It is assumed that problems using instances of this class will properly manage the solutions it contains.
    """

    def __init__(self, solutions: List[Solution]):
        super(CompositeSolution, self).__init__(
            len(solutions), len(solutions[0].objectives), len(solutions[0].constraints)
        )
        Check.is_not_none(solutions)
        Check.collection_is_not_empty(solutions)

        for solution in solutions:
            Check.that(
                len(solution.objectives) == len(solutions[0].objectives),
                "The solutions in the list must have the same number of objectives: "
                + str(len(solutions[0].objectives)),
            )
            Check.that(
                len(solution.constraints) == len(solutions[0].constraints),
                "The solutions in the list must have the same number of constraints: "
                + str(len(solutions[0].constraints)),
            )

        self.variables = solutions

    def __copy__(self):
        new_solution = CompositeSolution(self.variables)

        new_solution.objectives = self.objectives[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """Class representing permutation solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = PermutationSolution(len(self.variables), len(self.objectives), len(self.constraints))
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
