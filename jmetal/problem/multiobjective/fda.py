from abc import ABCMeta, ABC, abstractmethod
from math import sqrt, pow, sin, pi, cos, floor

from jmetal.util.observable import DefaultObservable

from jmetal.core.observable import Observable
from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: FDA
   :platform: Unix, Windows
   :synopsis: FDA problem family of dynamic multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class FDA(DynamicProblem, FloatProblem, ABC):
    def __init__(self, observable: Observable = DefaultObservable()):
        super(FDA, self).__init__()
        self.observable = observable
        self.tau_T = 5
        self.nT = 10
        self.time = 1.0
        self.problem_modified = False
        self.observable.register(self)

    def update(self, *args, **kwargs):
        counter: int = kwargs["COUNTER"]
        self.time = (1.0/self.nT) * floor(counter*1.0/self.tau_T)
        self.problem_modified = True

    def the_problem_has_changed(self) -> bool:
        self.problem_modified

    def reset(self) -> None:
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
        g = 1.0 + sum([pow(v - gT,2) for v in solution.variables[1:]])

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

    def __eval_g(self, solution: FloatSolution, lower_limit: int, upper_limit:int):
        g = sum([pow(v, 2) for v in solution.variables[lower_limit:upper_limit]])
        g += 1.0 + sum([pow(v + 1.0, 2.0) for v in solution.variables[upper_limit:]])

        return g

    def __eval_h(self, f: float, g: float) -> float:
        ht = 0.2 + 4.8 * pow(self.time, 2.0)
        return 1.0 - pow(f / g, ht)

    def get_name(self):
        return 'FDA2'

