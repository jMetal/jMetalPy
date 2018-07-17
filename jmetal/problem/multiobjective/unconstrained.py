from math import sqrt, exp, pow, sin

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: constrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for multi-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Kursawe(FloatProblem):
    """ Class representing problem Kursawe. """

    def __init__(self, number_of_variables: int=3, rf_path: str=None):
        super(Kursawe, self).__init__(rf_path=rf_path)
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-5.0 for _ in range(number_of_variables)]
        self.upper_bound = [5.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        fx = [0.0 for _ in range(self.number_of_objectives)]
        for i in range(self.number_of_variables - 1):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)
            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

        return solution

    def get_name(self):
        return 'Kursawe'


class Fonseca(FloatProblem):

    def __init__(self, rf_path: str=None):
        super(Fonseca, self).__init__(rf_path=rf_path)
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        n = self.number_of_variables
        solution.objectives[0] = 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))
        solution.objectives[1] = 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

        return solution

    def get_name(self):
        return 'Fonseca'


class Schaffer(FloatProblem):

    def __init__(self, rf_path: str=None):
        super(Schaffer, self).__init__(rf_path=rf_path)
        self.number_of_variables = 1
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-100000]
        self.upper_bound = [100000]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        value = solution.variables[0]

        solution.objectives[0] = value ** 2
        solution.objectives[1] = (value - 2) ** 2

        return solution

    def get_name(self):
        return 'Schaffer'


class Viennet2(FloatProblem):

    def __init__(self, rf_path: str=None):
        super(Viennet2, self).__init__(rf_path=rf_path)
        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x0 = solution.variables[0]
        x1 = solution.variables[1]

        f1 = (x0 - 2) * (x0 - 2) / 2.0 + (x1 + 1) * (x1 + 1) / 13.0 + 3.0
        f2 = (x0 + x1 - 3.0) * (x0 + x1 - 3.0) / 36.0 + (-x0 + x1 + 2.0) * (-x0 + x1 + 2.0) / 8.0 - 17.0
        f3 = (x0 + 2 * x1 - 1) * (x0 + 2 * x1 - 1) / 175.0 + (2 * x1 - x0) * (2 * x1 - x0) / 17.0 - 13.0

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3

        return solution

    def get_name(self):
        return 'Viennet2'
