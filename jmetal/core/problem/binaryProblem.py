from typing import TypeVar

from jmetal.core.problem.problem import Problem
from jmetal.core.solution.binarySolution import BinarySolution

S = TypeVar('S')

""" Class representing float problems """
__author__ = "Antonio J. Nebro"


class BinaryProblem(Problem[BinarySolution]):
    def __init__(self):
        pass

    def evaluate(self, solution: BinarySolution):
        pass

    def create_solution(self)->BinarySolution:
        pass




