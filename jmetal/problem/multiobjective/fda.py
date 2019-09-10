from abc import ABC, abstractmethod
from math import sqrt, pow, sin, pi, floor, cos

import numpy

from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: FDA
   :platform: Unix, Windows
   :synopsis: FDA problem family of dynamic multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class FDA(DynamicProblem, FloatProblem, ABC):
    def __init__(self):
        super(FDA, self).__init__()
        self.tau_T = 5
        self.nT = 10
        self.time = 1.0
        self.problem_modified = False

    def update(self, *args, **kwargs):
        counter: int = kwargs["COUNTER"]
        self.time = (1.0 / self.nT) * floor(counter * 1.0 / self.tau_T)
        self.problem_modified = True

    def the_problem_has_changed(self) -> bool:
        return self.problem_modified

    def clear_changed(self) -> None:
        self.problem_modified = False

    @abstractmethod
    def evaluate(self, solution: FloatSolution):
        pass


class FDA1(FDA):
    """ Problem FDA1.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(FDA1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def __eval_g(self, solution: FloatSolution):
        gT = sin(0.5 * pi * self.time)
        g = 1.0 + sum([pow(v - gT, 2) for v in solution.variables[1:]])

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'FDA1'


class FDA2(FDA):
    """ Problem FDA2

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 31.
    """

    def __init__(self, number_of_variables: int = 31):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(FDA2, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution, 1, len(solution.variables))
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def __eval_g(self, solution: FloatSolution, lower_limit: int, upper_limit: int):
        g = sum([pow(v, 2) for v in solution.variables[lower_limit:upper_limit]])
        g += 1.0 + sum([pow(v + 1.0, 2.0) for v in solution.variables[upper_limit:]])

        return g

    def __eval_h(self, f: float, g: float) -> float:
        ht = 0.2 + 4.8 * pow(self.time, 2.0)
        return 1.0 - pow(f / g, ht)

    def get_name(self):
        return 'FDA2'


class FDA3(FDA):
    """ Problem FDA3

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 30.
    """

    def __init__(self, number_of_variables: int = 30):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(FDA3, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.limitInfI = 0
        self.limitSupI = 1
        self.limitInfII = 1

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution, self.limitInfII)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = self.__eval_f(solution, self.limitInfI, self.limitSupI)
        solution.objectives[1] = g * h

        return solution

    def __eval_f(self, solution: FloatSolution, lower_limit: int, upper_limit: int):
        f = 0.0
        aux = 2.0 * sin(0.5 * pi * self.time)
        ft = pow(10, aux)
        f += sum([pow(v, ft) for v in solution.variables[lower_limit:upper_limit]])

        return f

    def __eval_g(self, solution: FloatSolution, lower_limit: int):
        gt = abs(sin(0.5 * pi * self.time))
        g = sum([pow(v - gt, 2) for v in solution.variables[lower_limit:]])
        g = g + 1.0 + gt

        return g

    def __eval_h(self, f: float, g: float) -> float:
        h = 1.0 - sqrt(f / g)
        return h

    def get_name(self):
        return 'FDA3'


class FDA4(FDA):
    """ Problem FDA4

    .. note:: Three-objective dynamic unconstrained problem. The default number of variables is 12.
    """
    M = 3

    def __init__(self, number_of_variables: int = 12):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(FDA4, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution, self.M - 1)

        solution.objectives[0] = self.__eval_f1(solution, g)
        solution.objectives[1] = self.__eval_fk(solution, g, 2)
        solution.objectives[2] = self.__eval_fm(solution, g)

        return solution

    def __eval_g(self, solution: FloatSolution, lower_limit: int):
        gt = abs(sin(0.5 * pi * self.time))
        g = sum([pow(v - gt, 2) for v in solution.variables[lower_limit:]])

        return g

    def __eval_f1(self, solution: FloatSolution, g: float) -> float:
        f = 1.0 + g
        mult = numpy.prod([cos(v * pi / 2.0) for v in solution.variables[:self.M - 1]])

        return f * mult

    def __eval_fk(self, solution: FloatSolution, g: float, k: int) -> float:
        f = 1.0 + g
        aux = sin((solution.variables[self.M - k] * pi) / 2.0)
        mult = numpy.prod([cos(v * pi / 2.0) for v in solution.variables[:self.M - k]])

        return f * mult * aux

    def __eval_fm(self, solution: FloatSolution, g: float) -> float:
        fm = 1.0 + g
        fm *= sin((solution.variables[0] * pi) / 2.0)

        return fm

    def get_name(self):
        return 'FDA4'


class FDA5(FDA):
    """ Problem FDA5

    .. note:: Three-objective dynamic unconstrained problem. The default number of variables is 12.
    """
    M = 3

    def __init__(self, number_of_variables: int = 12):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(FDA5, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution, self.M - 1)
        ft = 1.0 + 100.0 * pow(sin(0.5 * pi * self.time), 4.0)

        solution.objectives[0] = self.__eval_f1(solution, g, ft)
        solution.objectives[1] = self.__eval_fk(solution, g, 2, ft)
        solution.objectives[2] = self.__eval_fm(solution, g, ft)

        return solution

    def __eval_g(self, solution: FloatSolution, lower_limit: int):
        gt = abs(sin(0.5 * pi * self.time))
        g = sum([pow(v - gt, 2) for v in solution.variables[lower_limit:]])

        return g

    def __eval_f1(self, solution: FloatSolution, g: float, ft: float) -> float:
        f = 1.0 + g
        mult = numpy.prod([cos(pow(v, ft) * pi / 2.0) for v in solution.variables[:self.M - 1]])

        return f * mult

    def __eval_fk(self, solution: FloatSolution, g: float, k: int, ft: float) -> float:
        f = 1.0 + g

        mult = numpy.prod([cos(pow(v, ft) * pi / 2.0) for v in solution.variables[:self.M - k]])
        yy = pow(solution.variables[self.M - k], ft)
        mult *= sin(yy * pi / 2.0)

        return f * mult

    def __eval_fm(self, solution: FloatSolution, g: float, ft: float) -> float:
        fm = 1.0 + g
        y_1 = pow(solution.variables[0], ft)
        mult = sin(y_1 * pi / 2.0)

        return fm * mult

    def get_name(self):
        return 'FDA5'
