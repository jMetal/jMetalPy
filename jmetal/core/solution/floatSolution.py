from jmetal.core.solution.solution import Solution

""" Class representing float solutions """
__author__ = "Antonio J. Nebro"


class FloatSolution(Solution[float]):
    lower_bound = []
    upper_bound = []
    def __init__(self, number_of_variables:int, number_of_objectives: int):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives)

    '''
    def get_lower_bound(self, index: int) -> float:
        return self.lower_bound[index]

    def get_upper_bound(self, index: int) -> float:
        return self.upper_bound[index]

    '''