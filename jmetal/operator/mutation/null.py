import random

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.solution import Solution

""" Class implementing the null mutation operator """


class Null(MutationOperator[Solution]):
    def __init__(self):
        super(Null, self).__init__(probability=0)

    def execute(self, solution: Solution) -> Solution:
        return solution

