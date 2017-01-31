from typing import List

from jmetal.core.solution.solution import Solution

""" Class representing float solutions """
__author__ = "Antonio J. Nebro"

BitSet = List[bool]


class BinarySolution(Solution[BitSet]):
    def __init__(self, number_of_variables:int, number_of_objectives: int):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives)
        self.variables = [[] for x in range(number_of_variables)]

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total
