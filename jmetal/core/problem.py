import os
from os.path import dirname, join
from pathlib import Path
from typing import Generic, TypeVar
import random

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution
from jmetal.util.front_file import walk_up_folder, read_front_from_file_as_solutions

S = TypeVar('S')


class Problem(Generic[S]):
    """ Class representing problems. """

    def __init__(self):
        self.number_of_variables: int = None
        self.number_of_objectives: int = None
        self.number_of_constraints: int = None
        self.reference_front_path: str = 'problem/resources/{0}.pf'.format(self.get_name())

    def evaluate(self, solution: S) -> S:
        raise NotImplemented

    def evaluate_constraints(self, solution: S):
        raise NotImplemented

    def create_solution(self) -> S:
        raise NotImplemented

    def get_reference_front(self) -> list:
        front = []
        file_path = dirname(join(dirname(__file__)))
        computed_path = join(file_path, self.reference_front_path)

        if Path(computed_path).is_file():
            front = read_front_from_file_as_solutions(computed_path)

        return front

    def get_name(self) -> str:
        return self.__class__.__name__


class BinaryProblem(Problem[BinarySolution]):
    """ Class representing binary problems. """

    def evaluate(self, solution: BinarySolution) -> None:
        pass

    def create_solution(self) -> BinarySolution:
        pass


class FloatProblem(Problem[FloatSolution]):
    """ Class representing float problems. """

    def __init__(self):
        super().__init__()
        self.lower_bound: [] = None
        self.upper_bound: [] = None

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                     self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution


class IntegerProblem(Problem[IntegerSolution]):
    """ Class representing integer problems. """

    def __init__(self):
        super().__init__()
        self.lower_bound : [] = None
        self.upper_bound : [] = None

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound, self.upper_bound)

        new_solution.variables = \
            [int(random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0)) for i in range(self.number_of_variables)]

        return new_solution
