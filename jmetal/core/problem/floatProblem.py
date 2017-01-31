import random
from typing import TypeVar

from jmetal.core.problem.problem import Problem
from jmetal.core.solution.floatSolution import FloatSolution

S = TypeVar('S')

""" Class representing float problems """
__author__ = "Antonio J. Nebro"


class FloatProblem(Problem[FloatSolution]):
    def __init__(self):
        self.lower_bound = []
        self.upper_bound = []

    def evaluate(self, solution: FloatSolution):
        pass

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution


    '''
    def get_lower_bound(self, index: int) -> float:
        return self.lower_bound[index]

    def get_upper_bound(self, index: int) -> float:
        return self.upper_bound[index]

    def set_lower_bounds(self, lower_bound_list: []) -> None:
        self.lower_bound = lower_bound_list

    def set_upper_bounds(self, upper_bound_list: []) -> None:
        self.upper_bound = upper_bound_list

    '''



