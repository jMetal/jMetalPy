from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar, List
from pathlib import Path
import random

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution

S = TypeVar('S')


class Problem(Generic[S]):
    """ Class representing problems. """

    __metaclass__ = ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self, reference_front_path: str):
        self.number_of_variables: int = None
        self.number_of_objectives: int = None
        self.number_of_constraints: int = None

        self.obj_directions: List[int] = []
        self.obj_labels: List[str] = []

        self.reference_front: List[S] = None
        if reference_front_path:
            self.reference_front = self.read_front_from_file_as_solutions(reference_front_path)

    @staticmethod
    def read_front_from_file(file_path: str) -> List[List[float]]:
        """ Reads a front from a file and returns a list.

        :return: List of solution points. """
        front = []
        if Path(file_path).is_file():
            with open(file_path) as file:
                for line in file:
                    vector = [float(x) for x in line.split()]
                    front.append(vector)
        else:
            raise Exception('Reference front file was not found at {}'.format(file_path))

        return front

    @staticmethod
    def read_front_from_file_as_solutions(file_path: str) -> List[S]:
        """ Reads a front from a file and returns a list of solution objects.

        :return: List of solution objects. """
        front = []
        if Path(file_path).is_file():
            with open(file_path) as file:
                for line in file:
                    vector = [float(x) for x in line.split()]
                    solution = FloatSolution(2, 2, 0, [], [])
                    solution.objectives = vector

                    front.append(solution)
        else:
            raise Exception('Reference front file was not found at {}'.format(file_path))

        return front

    @abstractmethod
    def create_solution(self) -> S:
        """ Creates a random solution to the problem.

        :return: Solution. """
        pass

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """ Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be replaced.

        :return: Evaluated solution. """
        pass

    def evaluate_constraints(self, solution: S):
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class BinaryProblem(Problem[BinarySolution]):
    """ Class representing binary problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(BinaryProblem, self).__init__(reference_front_path=rf_path)

    def create_solution(self) -> BinarySolution:
        pass

    @abstractmethod
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        pass


class FloatProblem(Problem[FloatSolution]):
    """ Class representing float problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(FloatProblem, self).__init__(reference_front_path=rf_path)
        self.lower_bound = None
        self.upper_bound = None

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                     self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution

    @abstractmethod
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass


class IntegerProblem(Problem[IntegerSolution]):
    """ Class representing integer problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(IntegerProblem, self).__init__(reference_front_path=rf_path)
        self.lower_bound = None
        self.upper_bound = None

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

    @abstractmethod
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        pass
