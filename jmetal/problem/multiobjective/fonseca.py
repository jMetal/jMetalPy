from math import exp

from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution

""" Class representing Fonseca problem"""

class Fonseca(FloatProblem):
    def __init__(self):
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [ 4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        n = self.number_of_variables
        solution.objectives[0] = 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))
        solution.objectives[1] = 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

    def get_name(self):
        return "Fonseca"
