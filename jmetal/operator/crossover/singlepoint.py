import random
from typing import List

from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.solution.binarySolution import BinarySolution

""" Class implementing the binary single point crossover operator """
__author__ = "Antonio J. Nebro"


class SinglePoint(CrossoverOperator[BinarySolution, BinarySolution]):
    def __init__(self, probability: float):
        super(SinglePoint, self).__init__(probability=probability)

    def execute(self, solution_list: List[BinarySolution]) -> List[BinarySolution]:
        if len(solution_list) != 2:
            raise Exception("The number of parents is not two: " + str(len(solution_list)))

        offspring = [solution_list[0].deepclone(), solution_list[1].deepclone()]
        rand = random.random()
        if rand <= self.probability:
            pass

        return offspring

