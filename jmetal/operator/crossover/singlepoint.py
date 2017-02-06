import random
from typing import List

from jmetal.core.operator.crossoveroperator import CrossoverOperator
from jmetal.core.solution.binarySolution import BinarySolution
from copy import deepcopy

""" Class implementing the binary single point crossover operator """
__author__ = "Antonio J. Nebro"


class SinglePoint(CrossoverOperator[BinarySolution, BinarySolution]):
    def __init__(self, probability: float):
        super(SinglePoint, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: " + str(len(parents)))

        offspring = [deepcopy(parents[0]), deepcopy(parents[1])]
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
                offspring[0].variables[i] = deepcopy(parents[1].variables[i])
                offspring[1].variables[i] = deepcopy(parents[0].variables[i])

        return offspring

    def get_number_of_parents(self):
        return 2


