import random
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from jmetal.core.observer import Observer
from jmetal.core.solution import (
    BinarySolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
)
from jmetal.logger import get_logger

logger = get_logger(__name__)

S = TypeVar("S")


class Problem(Generic[S], ABC):
    """Class representing problems."""

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self):
        self.reference_front: List[S] = []
        self.directions: List[int] = []
        self.labels: List[str] = []

    @abstractmethod
    def number_of_variables(self) -> int:
        pass

    @abstractmethod
    def number_of_objectives(self) -> int:
        pass

    @abstractmethod
    def number_of_constraints(self) -> int:
        pass

    @abstractmethod
    def create_solution(self) -> S:
        """Creates a random_search solution to the problem.

        :return: Solution."""
        pass

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be replaced.
        Note that this framework ASSUMES minimization, thus solutions must be evaluated in consequence.

        :return: Evaluated solution."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class DynamicProblem(Problem[S], Observer, ABC):
    @abstractmethod
    def the_problem_has_changed(self) -> bool:
        pass

    @abstractmethod
    def clear_changed(self) -> None:
        pass


class BinaryProblem(Problem[BinarySolution], ABC):
    """Class representing binary problems."""

    def __init__(self):
        super(BinaryProblem, self).__init__()

        self.number_of_bits_per_variable = []

    def number_of_bits_per_variable_list(self):
        return self.number_of_bits_per_variable

    def total_number_of_bits(self):
        return sum(self.number_of_bits_per_variable)


class FloatProblem(Problem[FloatSolution], ABC):
    """Class representing float problems."""

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives(), self.number_of_constraints()
        )
        new_solution.variables = [
            random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
            for i in range(self.number_of_variables())
        ]

        return new_solution


class IntegerProblem(Problem[IntegerSolution], ABC):
    """Class representing integer problems."""

    def __init__(self):
        super(IntegerProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives(), self.number_of_constraints()
        )
        new_solution.variables = [
            round(random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0))
            for i in range(self.number_of_variables())
        ]

        return new_solution


class PermutationProblem(Problem[PermutationSolution], ABC):
    """Class representing permutation problems."""

    def __init__(self):
        super(PermutationProblem, self).__init__()


class OnTheFlyFloatProblem(FloatProblem):
    """ Class for defining float problems on the fly.

        Example:

        >>> # Defining problem Srinivas on the fly
        >>> def f1(x: [float]):
        >>>     return 2.0 + (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0)
        >>>
        >>> def f2(x: [float]):
        >>>     return 9.0 * x[0] - (x[1] - 1.0) * (x[1] - 1.0)
        >>>
        >>> def c1(x: [float]):
        >>>     return 1.0 - (x[0] * x[0] + x[1] * x[1]) / 225.0
        >>>
        >>> def c2(x: [float]):
        >>>     return (3.0 * x[1] - x[0]) / 10.0 - 1.0
        >>>
        >>> problem = OnTheFlyFloatProblem()\
            .set_name("Srinivas")\
            .add_variable(-20.0, 20.0)\
            .add_variable(-20.0, 20.0)\
            .add_function(f1)\
            .add_function(f2)\
            .add_constraint(c1)\
            .add_constraint(c2)
    """

    def __init__(self):
        super(OnTheFlyFloatProblem, self).__init__()
        self.functions = []
        self.constraints = []
        self.problem_name = None

    def set_name(self, name) -> "OnTheFlyFloatProblem":
        self.problem_name = name

        return self

    def add_function(self, function) -> "OnTheFlyFloatProblem":
        self.functions.append(function)

        return self

    def add_constraint(self, constraint) -> "OnTheFlyFloatProblem":
        self.constraints.append(constraint)

        return self

    def add_variable(self, lower_bound, upper_bound) -> "OnTheFlyFloatProblem":
        self.lower_bound.append(lower_bound)
        self.upper_bound.append(upper_bound)

        return self

    def number_of_objectives(self) -> int:
        return len(self.functions)

    def number_of_constraints(self) -> int:
        return len(self.constraints)

    def evaluate(self, solution: FloatSolution) -> None:
        for i in range(self.number_of_objectives()):
            solution.objectives[i] = self.functions[i](solution.variables)

        for i in range(self.number_of_constraints()):
            solution.constraints[i] = self.constraints[i](solution.variables)

    def name(self) -> str:
        return self.problem_name
