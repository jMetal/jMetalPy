from math import sqrt, exp, pow, sin
from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution
import random

""" Class representing problem Kursawe """
__author__ = "Antonio J. Nebro"


class Kursawe(FloatProblem):
    def __init__(self, number_of_variables: int = 3):
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        upper_bound = [5.0 for i in range(number_of_variables)]
        lower_bound = [-5.0 for i in range(number_of_variables)]

        self.set_lower_bounds(lower_bound)
        self.set_upper_bounds(upper_bound)


    def evaluate(self, solution: FloatSolution):
        '''

        :param solution:
        :return:
        '''

        fx = [0.0 for x in range(self.number_of_objectives)]
        for i in range(self.number_of_variables):
            xi = solution.variable[i] * solution.variable[i]
            xj = solution.variable[i + 1] * solution.variable[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)

            fx[1] += pow(abs(solution.variable[i]), 0.8) + 5.0 * sin(pow(solution.variable[i], 3.0))

            solution.set_objective(0, fx[0])
            solution.set_objective(1, fx[1])

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_objectives, self.number_of_variables, self.lower_bound, self.upper_bound)
        new_solution.variable = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for x in range(self.number_of_variables)
             for i in range(self.number_of_variables)]

        return new_solution

