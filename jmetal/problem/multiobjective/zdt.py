"""
    ============================================================
    :mod:`zdt` -- ZDT problem family of multi-objective problems
    ============================================================

    This module provides the implementation of the ZDT problems defined in:

    Zitzler, E., Deb, K., Thiele, L.
    Comparison of multiobjective evolutionary algorithms: Empirical results.
    Evolutionary Computation 8(2), pp: 173-195. 2000

    DOI: https://doi.org/10.1162/106365600568202


    These problems are scalable in the number of the decision variables, i.e. the Pareto front
    is the same independently of the number of variables

    .. module:: jmetal.problem.multiobjective.zdt

    :platform: Unix, Windows
    :synopsis: Module including the implementation of the ZDT problem family.

    .. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>

"""


from math import sqrt, exp, pow, sin, pi, cos

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class ZDT1(FloatProblem):
    """ Problem ZDT1

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.

    .. note:: Continuous problem having a convex Pareto front

    """
    def __init__(self, number_of_variables: int = 30):
        """Constructor

        Arguments:

        :param number_of_variables: number of decision variables of the problem
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)
        g = constant * g
        g = g + 1.0
        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return "ZDT1"


class ZDT2(FloatProblem):
    """ Problem ZDT2

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.

    .. note:: Continuous problem having a non-convex Pareto front

    """
    def __init__(self, number_of_variables: int = 30):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)
        g = constant * g
        g = g + 1.0
        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def get_name(self):
        return "ZDT2"


class ZDT3(FloatProblem):
    """ Problem ZDT3

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.

    .. note:: Continuous problem having a partitioned Pareto front

    """
    def __init__(self, number_of_variables: int = 30):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)
        g = constant * g
        g = g + 1.0
        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g) - (f / g) * sin(10.0 * f * pi)

    def get_name(self):
        return "ZDT3"


class ZDT4(FloatProblem):
    """ Problem ZDT4

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.

    .. note:: Continuous multi-modal problem having a convex Pareto front

    """
    def __init__(self, number_of_variables: int = 10):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-5.0]
        self.upper_bound = self.number_of_variables * [5.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = 0.0

        for i in range(1, solution.number_of_variables):
            g += pow(solution.variables[i], 2.0) - 10.0 * cos(4.0 * pi * solution.variables[i])

        g += 1.0 + 10.0 * (solution.number_of_variables - 1)

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return "ZDT4"


class ZDT6(FloatProblem):
    """ Problem ZDT6

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.

    .. note:: Continuous problem having a non-convex Pareto front

    """
    def __init__(self, number_of_variables: int = 10):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]
        g = g / (solution.number_of_variables - 1)
        g = pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def get_name(self):
        return "ZDT6"
