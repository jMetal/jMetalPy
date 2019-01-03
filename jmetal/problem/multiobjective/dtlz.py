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

    def __init__(self, number_of_variables: int = 7, number_of_objectives=3):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        super(DTLZ1, self).__init__()
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


class DTLZ2(DTLZ1):
    """ Problem DTLZ2. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, number_of_variables: int = 12, number_of_objectives=3):
        """:param number_of_variables: number of decision variables of the problem
        """
        super(DTLZ2, self).__init__(number_of_variables, number_of_objectives)

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


class DTLZ3(DTLZ1):
    """ Problem DTLZ3. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, number_of_variables: int = 12, number_of_objectives=3):
        """:param number_of_variables: number of decision variables of the problem
        """
        super(DTLZ3, self).__init__(number_of_variables, number_of_objectives)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) ** 2 - cos(20.0 * pi * (x - 0.5)) for x in solution.variables[self.number_of_variables - k:]])
        g = 100.0 * (k + g)

        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(solution.variables[j] * 0.5 * pi)

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(solution.variables[aux] * 0.5 * pi)

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ3'


class DTLZ4(DTLZ1):
    """ Problem DTLZ4. Continuous problem having a convex Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 12 and 3.
    """

    def __init__(self, number_of_variables: int = 12, number_of_objectives=3):
        """:param number_of_variables: number of decision variables of the problem
        """
        super(DTLZ4, self).__init__(number_of_variables, number_of_objectives)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        alpha = 100.0
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) ** 2 for x in solution.variables[self.number_of_variables - k:]])
        f = [1.0 + g for _ in range(self.number_of_objectives)]

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                f[i] *= cos(pow(solution.variables[j], alpha) * pi / 2.0)

            if i != 0:
                aux = self.number_of_objectives - (i + 1)
                f[i] *= sin(pow(solution.variables[aux], alpha) * pi / 2.0)

        solution.objectives = [f[x] for x in range(self.number_of_objectives)]

        return solution

    def get_name(self):
        return 'DTLZ4'

