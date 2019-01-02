import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List
import random

from jmetal.core.observable import Observer
from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution

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

        self.reference_front: List[S] = None

        self.directions: List[int] = []
        self.labels: List[str] = []

    @abstractmethod
    def create_solution(self) -> S:
        """ Creates a random solution to the problem.

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

    def create_solution(self) -> BinarySolution:
        pass


class FloatProblem(Problem[FloatSolution], ABC):
    """ Class representing float problems. """

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.lower_bound,
            self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution


class OnTheFlyFloatProblem(FloatProblem):

    def __init__(self):
        super(OnTheFlyFloatProblem, self).__init__()
        self.objective_functions = []
        self.constraints = []
        self.name = ""

    def set_name(self, name):
        self.name = name

        return self

    def add_function(self, function):
        self.objective_functions.append(function)
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

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        for i in range(self.number_of_objectives):
            solution.objectives[i] = self.objective_functions[i](solution.variables)

        if self.number_of_constraints > 0:
            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in self.constraints:
                violation_degree = constrain(solution.variables)
                if violation_degree < 0.0:
                    overall_constraint_violation += violation_degree
                    number_of_violated_constraints += 1

            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

    def get_name(self) -> str:
        return self.name


class IntegerProblem(Problem[IntegerSolution], ABC):
    """ Class representing integer problems. """

    def __init__(self):
        super(IntegerProblem, self).__init__()
        self.lower_bound = None
        self.upper_bound = None

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.lower_bound,
            self.upper_bound)

        new_solution.variables = \
            [int(random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0))
             for i in range(self.number_of_variables)]

        return new_solution



