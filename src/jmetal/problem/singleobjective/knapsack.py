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
    """Class representing Knapsack Problem."""

    def __init__(
        self,
        number_of_items: int = 50,
        capacity: float = 1000,
        weights: list = None,
        profits: list = None,
        from_file: bool = False,
        filename: str = None,
    ):
        super(Knapsack, self).__init__()

        if from_file:
            self.__read_from_file(filename)
        else:
            self.capacity = capacity
            self.weights = weights
            self.profits = profits
            self.number_of_bits = number_of_items

        self.obj_directions = [self.MAXIMIZE]

    def number_of_variables(self) -> int:
        return self.number_of_bits

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 1

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
            raise FileNotFoundError("Filename can not be None")

        with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]

            self.number_of_bits = int(data[0][0])
            self.capacity = float(data[1][0])

            weights_and_profits = np.asarray(data[2:], dtype=np.float32)

            self.weights = weights_and_profits[:, 0]
            self.profits = weights_and_profits[:, 1]

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_profits = 0.0
        total_weights = 0.0

        # Access the bits from variables
        bits = solution.variables
        for i in range(len(bits)):
            if bits[i]:
                total_profits += self.profits[i]
                total_weights += self.weights[i]

        if total_weights > self.capacity:
            total_profits = 0.0

        solution.objectives[0] = -1.0 * total_profits
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_bits,
            number_of_objectives=self.number_of_objectives()
        )
        
        # The BinarySolution initializes with empty variables, we need to set the bits
        # The bits will be stored in variables[0] as a numpy array
        new_solution.variables[0] = np.random.choice([True, False], size=self.number_of_bits)
        return new_solution

    def name(self):
        return "Knapsack"
