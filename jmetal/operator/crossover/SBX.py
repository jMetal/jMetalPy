import random
from typing import List

from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.solution.binarySolution import BinarySolution
from copy import deepcopy

from jmetal.core.solution.floatSolution import FloatSolution

""" Class implementing the binary single point crossover operator """
__author__ = "Antonio J. Nebro"


class SBX(CrossoverOperator[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14
    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBX, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: " + str(len(parents)))

        offspring = [deepcopy(parents[0]), deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1 = parents[0].variables[i]
                value_x2 = parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1 = value_x1
                            y2 = value_x2
                        else:
                            y1 = value_x2
                            y2 = value_x1

                        lowerBound = parents[0].lower_bound[i]
                        upperBound = parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lowerBound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if (rand <= (1.0 / alpha)):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upperBound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq =  pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1));

                        #c1 = solutionRepair.repairSolutionVariableValue(c1, lowerBound, upperBound);
                        #c2 = solutionRepair.repairSolutionVariableValue(c2, lowerBound, upperBound);
                        if c1 < lowerBound:
                            c1 = lowerBound
                        if c2 < lowerBound:
                            c2 = lowerBound
                        if c1 > upperBound:
                            c1 = upperBound
                        if c2 > upperBound:
                            c2 = upperBound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

    def get_number_of_parents(self):
        return 2


