import random
from math import sqrt, exp, pow, sin

from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution

""" Class representing problem Sphere """
__author__ = "Antonio J. Nebro"


class Sphere(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.lower_bound = [-5.12 for i in range(number_of_variables)]
        self.upper_bound = [5.12 for i in range(number_of_variables)]

    def evaluate(self, solution: FloatSolution):
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)]

        return new_solution

    def get_name(self):
        return "Sphere"
