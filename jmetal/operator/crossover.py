import copy
import random
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution, PermutationSolution

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):

    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Null crossover'


class PMXCrossover(Crossover[PermutationSolution, PermutationSolution]):

    def __init__(self, probability: float):
        super(PMXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                size = min(len(offspring[0].variables[i]), len(offspring[1].variables[i]))
                p1, p2 = [0] * size, [0] * size

                # Initialize the position of each indices in the individuals
                for j in range(size):
                    p1[offspring[0].variables[i][j]] = j
                    p2[offspring[1].variables[i][j]] = j

                # Choose crossover points
                cxpoint1 = random.randint(0, size)
                cxpoint2 = random.randint(0, size - 1)

                if cxpoint2 >= cxpoint1:
                    cxpoint2 += 1
                else:  # Swap the two cx points
                    cxpoint1, cxpoint2 = cxpoint2, cxpoint1

                # Apply crossover between cx points
                for j in range(cxpoint1, cxpoint2):
                    # Keep track of the selected values
                    temp1 = offspring[0].variables[i][j]
                    temp2 = offspring[1].variables[i][j]

                    # Swap the matched value
                    offspring[0].variables[i][j], offspring[0].variables[i][p1[temp2]] = temp2, temp1
                    offspring[1].variables[i][j], offspring[1].variables[i][p2[temp1]] = temp1, temp2

                    # Position bookkeeping
                    p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                    p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

                offspring[0].variables[i] = p1
                offspring[1].variables[i] = p2

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Partially Matched crossover'


class CXCrossover(Crossover[PermutationSolution, PermutationSolution]):

    def __init__(self, probability: float):
        super(CXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[1]), copy.deepcopy(parents[0])]
        rand = random.random()

        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                idx = random.randint(0, len(parents[0].variables[i]) - 1)
                curr_idx = idx
                cycle = []

                while True:
                    cycle.append(curr_idx)
                    curr_idx = parents[0].variables[i].index(parents[1].variables[i][curr_idx])

                    if curr_idx == idx:
                        break

                for j in range(len(parents[0].variables[i])):
                    if j in cycle:
                        offspring[0].variables[i][j] = parents[0].variables[i][j]
                        offspring[1].variables[i][j] = parents[0].variables[i][j]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Cycle crossover'


class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
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

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'SBX crossover'


class SPXCrossover(Crossover[BinarySolution, BinarySolution]):

    def __init__(self, probability: float):
        super(SPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
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
            bitset1 = copy.copy(parents[0].variables[variable_to_cut])
            bitset2 = copy.copy(parents[1].variables[variable_to_cut])

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

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Single point crossover'


class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """ This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    """

    def __init__(self, CR: float, F: float, K: float):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Differential Evolution crossover'
