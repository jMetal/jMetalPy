from jmetal.core.solution.solution import Solution

""" Class representing integer solutions """

class IntegerSolution(Solution[int]):
    lower_bound = []
    upper_bound = []
    def __init__(self, number_of_variables:int, number_of_objectives: int):
        super(IntegerSolution, self).__init__(number_of_variables, number_of_objectives)
