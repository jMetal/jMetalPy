import random
from math import cos, pi, pow, sin, sqrt, exp

from jmetal.core.problem import FloatProblem, BinaryProblem
from jmetal.core.solution import FloatSolution, BinarySolution

"""
.. module:: ZDT
   :platform: Unix, Windows
   :synopsis: ZDT problem family of multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ZDT1(FloatProblem):
    """Problem ZDT1.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT1, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = number_of_variables * [0.0]
        self.upper_bound = number_of_variables * [1.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.eval_g(solution)
        h = self.eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (len(solution.variables) - 1)

        return constant * g + 1.0

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def name(self):
        return "ZDT1"


class ZDT1Modified(ZDT1):
    """Problem ZDT1Modified.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """

    def __init__(self, number_of_variables=30):
        super(ZDT1Modified, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        s: float = 0.0
        for i in range(1000):
            for j in range(10000):
                s += i * 0.235 / 1.234 + 1.23525 * j
        return super().evaluate(solution)


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
    """Problem ZDT2.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def name(self):
        return "ZDT2"


class ZDT3(ZDT1):
    """Problem ZDT3.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a partitioned Pareto front
    """

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g) - (f / g) * sin(10.0 * f * pi)

    def name(self):
        return "ZDT3"


class ZDT4(ZDT1):
    """Problem ZDT4.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous multi-modal problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int = 10):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT4, self).__init__()
        self.lower_bound = number_of_variables * [-5.0]
        self.upper_bound = number_of_variables * [5.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def eval_g(self, solution: FloatSolution):
        g = 0.0

        for i in range(1, len(solution.variables)):
            g += pow(solution.variables[i], 2.0) - 10.0 * cos(4.0 * pi * solution.variables[i])

        g += 1.0 + 10.0 * (len(solution.variables) - 1)

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def name(self):
        return "ZDT4"


class ZDT5(BinaryProblem):
    """Problem ZDT5.

    .. note:: Bi-objective binary unconstrained problem. The default number of variables is 11.
    """

    def __init__(self, number_of_variables: int = 11):
        """:param number_of_bits: Number of bits of each variable of the problem."""
        super(ZDT5, self).__init__()

        self.number_of_bits_per_variable = [5 for _ in range(0, number_of_variables)]
        self.number_of_bits_per_variable[0] = 30

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

    def number_of_variables(self) -> int:
        return len(self.number_of_bits_per_variable)

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        solution.objectives[0] = 1.0 + solution.cardinality(0)

        g = self.eval_g(solution)
        h = 1.0 / solution.objectives[0]

        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: BinarySolution):
        result = 0.0
        for i in range(1, len(solution.variables)):
            result = result + self.eval_v(solution.cardinality(i))

        return result

    def eval_v(self, value):
        if value < 5.0:
            return 2.0 + value
        else:
            return 1.0

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables(), number_of_objectives=2)
        for i in range(self.number_of_variables()):
            new_solution.variables[i] = [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits_per_variable[i])]
        return new_solution

    def name(self):
        return "ZDT5"


class ZDT6(ZDT1):
    """Problem ZDT6.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def __init__(self, number_of_variables: int = 10):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT6, self).__init__(number_of_variables=number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = (
            1.0 - exp(-4.0 * solution.variables[0]) * (sin(6.0 * pi * solution.variables[0])) ** 6.0
        )

        g = self.eval_g(solution)
        h = self.eval_h(solution.objectives[0], g)
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]
        g = g / (len(solution.variables) - 1)
        g = pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def name(self):
        return "ZDT6"
