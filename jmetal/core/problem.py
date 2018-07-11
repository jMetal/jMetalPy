from abc import ABCMeta, abstractmethod
from os.path import dirname, join
from pathlib import Path
from typing import Generic, TypeVar
import random

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution
from jmetal.util.front_file import read_front_from_file_as_solutions

S = TypeVar('S')


class Problem(Generic[S]):
    """ Class representing problems. """

    __metaclass__ = ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self):
        self.number_of_variables = None
        self.number_of_objectives = None
        self.number_of_constraints = None
        self.obj_directions = []

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """ Evaluate a solution.

        :return: Evaluated solution. """
        pass

    @abstractmethod
    def create_solution(self) -> S:
        """ Creates a random solution to the problem.

        :return: Solution. """
        pass

    def evaluate_constraints(self, solution: S):
        pass

    def get_reference_front(self) -> list:
        """ Get the reference front to the problem (if any).
        This method read front files (.pf) located in `jmetal/problem/reference_front/`, which must have the same
        name as the problem.

        :return: Front."""
        reference_front_path = 'problem/reference_front/{0}.pf'.format(self.get_name())

        front = []
        file_path = dirname(join(dirname(__file__)))
        computed_path = join(file_path, reference_front_path)

        if Path(computed_path).is_file():
            front = read_front_from_file_as_solutions(computed_path)

        return front

    def get_name(self) -> str:
        return self.__class__.__name__


class BinaryProblem(Problem[BinarySolution]):
    """ Class representing binary problems. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        pass

    @abstractmethod
    def create_solution(self) -> BinarySolution:
        pass


class FloatProblem(Problem[FloatSolution]):
    """ Class representing float problems. """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.lower_bound = None
        self.upper_bound = None

    @abstractmethod
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                     self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution


class IntegerProblem(Problem[IntegerSolution]):
    """ Class representing integer problems. """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(IntegerProblem, self).__init__()
        self.lower_bound = None
        self.upper_bound = None

    @abstractmethod
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        pass

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound, self.upper_bound)

        new_solution.variables = \
            [int(random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0))
             for i in range(self.number_of_variables)]

        return new_solution
