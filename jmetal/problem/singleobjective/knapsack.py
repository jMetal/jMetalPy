import random

import numpy as np

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

"""
.. module:: knapsack
   :platform: Unix, Windows
   :synopsis: Single Objective Knapsack problem

.. moduleauthor:: Alejandro Marrero <alu0100825008@ull.edu.es>
"""


class Knapsack(BinaryProblem):
    """ Class representing Knapsack Problem. """

    def __init__(self, number_of_items: int = 50, capacity: float = 1000, weights: list = None,
                 profits: list = None, from_file: bool = False, filename: str = None):
        super(Knapsack, self).__init__()

        if from_file:
            self.__read_from_file(filename)
        else:
            self.capacity = capacity
            self.weights = weights
            self.profits = profits
            self.number_of_bits = number_of_items

        self.number_of_variables = 1
        self.obj_directions = [self.MAXIMIZE]
        self.number_of_objectives = 1
        self.number_of_constraints = 1

    def __read_from_file(self, filename: str):
        """
        This function reads a Knapsack Problem instance from a file.
        It expects the following format:

            num_of_items (dimension)
            capacity of the knapsack
            num_of_items-tuples of weight-profit

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]

            self.number_of_bits = int(data[0][0])
            self.capacity = float(data[1][0])

            weights_and_profits = np.asfarray(data[2:], dtype=np.float32)

            self.weights = weights_and_profits[:, 0]
            self.profits = weights_and_profits[:, 1]

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_profits = 0.0
        total_weigths = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_profits += self.profits[index]
                total_weigths += self.weights[index]

        if total_weigths > self.capacity:
            total_profits = 0.0

        solution.objectives[0] = -1.0 * total_profits
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(
                self.number_of_bits)]

        return new_solution

    def get_name(self):
        return 'Knapsack'
