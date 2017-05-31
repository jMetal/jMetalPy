from math import sqrt, exp, pow, sin
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

__author__ = "Antonio J. Nebro"


class Kursawe(FloatProblem):
    """ Class representing problem Kursawe """

    def __init__(self, number_of_variables: int = 3):
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0
        self.lower_bound = [-5.0 for i in range(number_of_variables)]
        self.upper_bound = [5.0 for i in range(number_of_variables)]

<<<<<<< HEAD:jmetal/problem/multiobjective/kursawe.py
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
=======
    def evaluate(self, solution: FloatSolution) -> None:
>>>>>>> 0c3a3b5ecb116c4ec22fd8540d233f554fdd700a:jmetal/problem/multiobjective.py
        fx = [0.0 for x in range(self.number_of_objectives)]
        for i in range(self.number_of_variables):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)
            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))
<<<<<<< HEAD:jmetal/problem/multiobjective/kursawe.py

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

    def get_name(self):
        return "Kursawe"
=======
            solution.objectives[0] = fx[0]
            solution.objectives[1] = fx[1]

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.lower_bound, self.upper_bound)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for x in range(self.number_of_variables)
             for i in range(self.number_of_variables)]
        return new_solution
>>>>>>> 0c3a3b5ecb116c4ec22fd8540d233f554fdd700a:jmetal/problem/multiobjective.py
