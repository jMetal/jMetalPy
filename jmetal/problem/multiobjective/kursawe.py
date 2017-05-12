from math import sqrt, exp, pow, sin

from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution

""" Class representing problem Kursawe """
__author__ = "Antonio J. Nebro"


class Kursawe(FloatProblem):
    def __init__(self, number_of_variables = 3):
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.lower_bound = [-5.0 for i in range(number_of_variables)]
        self.upper_bound = [5.0 for i in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        fx = [0.0 for x in range(self.number_of_objectives)]
        for i in range(self.number_of_variables):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)

            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

    def get_name(self):
        return "Kursawe"
