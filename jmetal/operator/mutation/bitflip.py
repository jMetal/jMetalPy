import random

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.binarySolution import BinarySolution

""" Class implementing the binary BitFlip mutation operator """
__author__ = "Antonio J. Nebro"


class BitFlip(MutationOperator[BinarySolution]):
    def __init__(self, probability: float):
        super(BitFlip, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                print(rand, self.probability)
                if rand <= self.probability:
                    solution.variables[i][j] = True if solution.variables[i][j] == False else False

        return solution

