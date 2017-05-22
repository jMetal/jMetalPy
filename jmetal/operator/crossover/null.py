import copy
from typing import List

from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.solution.solution import Solution

""" Class implementing the null crossover operator """

class Null(CrossoverOperator[Solution, Solution]):
    def __init__(self):
        super(Null, self).__init__(probability=0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: " + str(len(parents)))

        return [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]

    def get_number_of_parents(self):
        return 2


