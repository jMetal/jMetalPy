import logging
import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

from jmetal.core.observer import Observer
from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution, PermutationSolution

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')


class Problem(Generic[S], ABC):
    """ Class representing problems. """

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self):
        self.number_of_variables: int = 0
        self.number_of_objectives: int = 0
        self.number_of_constraints: int = 0

        self.reference_front: List[S] = []

        self.directions: List[int] = []
        self.labels: List[str] = []

    @abstractmethod
    def create_solution(self) -> S:
        """ Creates a random_search solution to the problem.

        :return: Solution. """
        pass

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """ Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be
        replaced. Note that this framework ASSUMES minimization, thus solutions must be evaluated in consequence.

        :return: Evaluated solution. """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class DynamicProblem(Problem[S], Observer, ABC):

    @abstractmethod
    def the_problem_has_changed(self) -> bool:
        pass

    @abstractmethod
    def clear_changed(self) -> None:
        pass


class BinaryProblem(Problem[BinarySolution], ABC):
    """ Class representing binary problems. """

    def __init__(self):
        super(BinaryProblem, self).__init__()


class FloatProblem(Problem[FloatSolution], ABC):
    """ Class representing float problems. """

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in
             range(self.number_of_variables)]

        return new_solution


class IntegerProblem(Problem[IntegerSolution], ABC):
    """ Class representing integer problems. """

    def __init__(self):
        super(IntegerProblem, self).__init__()
        self.lower_bound = None
        self.upper_bound = None

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = \
            [int(random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0))
             for i in range(self.number_of_variables)]

        return new_solution


class PermutationProblem(Problem[PermutationSolution], ABC):
    """ Class representing permutation problems. """

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
        self.name = None

    def set_name(self, name):
        self.name = name

        return self

    def add_function(self, function):
        self.functions.append(function)
        self.number_of_objectives += 1

        return self

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        self.number_of_constraints += 1

        return self

    def add_variable(self, lower_bound, upper_bound):
        self.lower_bound.append(lower_bound)
        self.upper_bound.append(upper_bound)
        self.number_of_variables += 1

        return self

    def evaluate(self, solution: FloatSolution):
        for i in range(self.number_of_objectives):
            solution.objectives[i] = self.functions[i](solution.variables)

        for i in range(self.number_of_constraints):
            solution.constraints[i] = self.constraints[i](solution.variables)

    def get_name(self) -> str:
        return self.name
