from math import pi, cos, sin

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: DTLZ
   :platform: Unix, Windows
   :synopsis: DTLZ problem family of multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class DTLZ1(FloatProblem):
    """ Problem DTLZ1. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 7 and 3.
    """

    def __init__(self, number_of_variables: int = 7, number_of_objectives=3, rf_path: str=None):
        """ :param number_of_variables: number of decision variables of the problem.
        :param rf_path: Path to the reference front file (if any). Default to None.
        """
        super(DTLZ1, self).__init__(rf_path=rf_path)
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in solution.variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        solution.objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= solution.variables[j]

            if i != 0:
                solution.objectives[i] *= 1 - solution.variables[self.number_of_objectives - (i + 1)]

        return solution

    def get_name(self):
        return 'DTLZ1'


class DTLZ2(FloatProblem):
    """ Problem DTLZ2. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, number_of_variables: int = 12, number_of_objectives=3, rf_path: str=None):
        """:param number_of_variables: number of decision variables of the problem
        :param rf_path: Path to the reference front file (if any). Default to None.
        """
        super(DTLZ2, self).__init__(rf_path=rf_path)
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(number_of_objectives)]

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) for x in solution.variables[self.number_of_variables - k:]])

        solution.objectives = [1.0 + g] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                solution.objectives[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                solution.objectives[i] *= sin(0.5 * pi * solution.variables[self.number_of_objectives - (i + 1)])

        return solution

    def get_name(self):
        return 'DTLZ2'
