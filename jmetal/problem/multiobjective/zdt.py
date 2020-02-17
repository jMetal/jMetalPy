from math import sqrt, pow, sin, pi, cos

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: ZDT
   :platform: Unix, Windows
   :synopsis: ZDT problem family of multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ZDT1(FloatProblem):
    """ Problem ZDT1.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int=30):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(ZDT1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['x', 'y']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.eval_g(solution)
        h = self.eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)

        return constant * g + 1.0

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'ZDT1'


class ZDT1Modified(ZDT1):
    """ Problem ZDT1Modified.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """
    def __init__(self, number_of_variables = 30):
        super(ZDT1Modified, self).__init__(number_of_variables)

    def evaluate(self, solution:FloatSolution) -> FloatSolution:
        s: float = 0.0
        for i in range(1000):
            for j in range(10000):
                s += i * 0.235 / 1.234 + 1.23525 * j
        return super().evaluate(solution)


class ZDT2(ZDT1):
    """ Problem ZDT2.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def get_name(self):
        return 'ZDT2'


class ZDT3(ZDT1):
    """ Problem ZDT3.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a partitioned Pareto front
    """
    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g) - (f / g) * sin(10.0 * f * pi)

    def get_name(self):
        return 'ZDT3'


class ZDT4(ZDT1):
    """ Problem ZDT4.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous multi-modal problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int=10):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(ZDT4, self).__init__(number_of_variables=number_of_variables)
        self.lower_bound = self.number_of_variables * [-5.0]
        self.upper_bound = self.number_of_variables * [5.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def eval_g(self, solution: FloatSolution):
        g = 0.0

        for i in range(1, solution.number_of_variables):
            g += pow(solution.variables[i], 2.0) - 10.0 * cos(4.0 * pi * solution.variables[i])

        g += 1.0 + 10.0 * (solution.number_of_variables - 1)

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'ZDT4'


class ZDT6(ZDT1):
    """ Problem ZDT6.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def __init__(self, number_of_variables: int=10):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(ZDT6, self).__init__(number_of_variables=number_of_variables)

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]
        g = g / (solution.number_of_variables - 1)
        g = pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def get_name(self):
        return 'ZDT6'
