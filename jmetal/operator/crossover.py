import copy
import random
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):

    def __init__(self):
        super(NullCrossover, self).__init__(probability=0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        return parents

    def get_number_of_parents(self):
        return 2

    def get_name(self):
        return 'Null crossover'


class SBX(Crossover[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBX, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.copy(parents[0]), copy.copy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

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

    def get_name(self):
        return 'SBX crossover'


class SP(Crossover[BinarySolution, BinarySolution]):

    def __init__(self, probability: float):
        super(SP, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.copy(parents[0]), copy.copy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].get_total_number_of_bits()

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(0, total_number_of_bits)

            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[variable_to_cut])

            # 4. Compute the bit into the selected variable
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[variable_to_cut]) - diff

            # 5. Apply the crossover to the variable
            bitset1 = parents[0].variables[variable_to_cut]
            bitset2 = parents[1].variables[variable_to_cut]

            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap

            offspring[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[variable_to_cut] = bitset2

            # 6. Apply the crossover to the other variables
            for i in range(variable_to_cut + 1, parents[0].number_of_variables):
                offspring[0].variables[i] = copy.deepcopy(parents[1].variables[i])
                offspring[1].variables[i] = copy.deepcopy(parents[0].variables[i])

        return offspring

    def get_number_of_parents(self):
        return 2

    def get_name(self):
        return 'Single point crossover'
