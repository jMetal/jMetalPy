from jmetal.core.solution.solution import Solution

""" Class representing float solutions """
__author__ = "Antonio J. Nebro"


class FloatSolution(Solution[float]):
    def __init__(self):
        self.lower_bound = []
        self.upper_bound = []

    def get_lower_bound(self, index: int) -> float:
        return self.lower_bound[index]

    def get_upper_bound(self, index: int) -> float:
        return self.upper_bound[index]
'''
    def set_upper_bound(self, index: int, value: float):
        self.upper_bound[index] = value

    def set_lower_bound(self, index: int, value: float):
        self.lower_bound[index] = value

'''
