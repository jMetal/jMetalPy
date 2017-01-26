from jmetal.core.solution.solution import Solution

""" Class representing float solutions """
__author__ = "Antonio J. Nebro"


class FloatSolution(Solution[float]):
    def __init__(self, number_of_variables:int, number_of_objectives: int, lower_bound = [], upper_bound=[]):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self, index: int) -> float:
        return self.lower_bound[index]

    def get_upper_bound(self, index: int) -> float:
        return self.upper_bound[index]
