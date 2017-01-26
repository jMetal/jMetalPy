from typing import TypeVar, Generic
from jmetal.core.solution.floatSolution import FloatSolution

S = TypeVar('S')

""" Class representing float problems """
__author__ = "Antonio J. Nebro"


class FloatProblem(FloatSolution):
    def __init__(self):
        pass

    def evaluate(self, solution: FloatSolution):
        pass

    def create_solution(self)->FloatSolution:
        pass

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



