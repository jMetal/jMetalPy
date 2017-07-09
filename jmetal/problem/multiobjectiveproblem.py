from math import sqrt, exp, pow, sin, pi, cos

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

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        fx = [0.0 for x in range(self.number_of_objectives)]

        fx[0] = 0.0
        for i in range(self.number_of_variables - 1):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)

        fx[1] = 0.0
        for i in range(self.number_of_variables):
            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

    def get_name(self):
        return "Kursawe"


class Fonseca(FloatProblem):
    def __init__(self):
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [ 4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        n = self.number_of_variables
        solution.objectives[0] = 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))
        solution.objectives[1] = 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

    def get_name(self):
        return "Fonseca"


class Schaffer(FloatProblem):
    def __init__(self):
        self.number_of_variables = 1
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = [-100000]
        self.upper_bound = [100000]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        value = solution.variables[0]

        solution.objectives[0] = value**2
        solution.objectives[1] = (value-2)**2

    def get_name(self):
        return "Schaffer"


class Viennet2(FloatProblem):
    def __init__(self):
        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        x0 = solution.variables[0]
        x1 = solution.variables[1]

        f1 = (x0 - 2) * (x0 - 2) / 2.0 + (x1 + 1) * (x1 + 1) / 13.0 + 3.0
        f2 = (x0 + x1 - 3.0) * (x0 + x1 - 3.0) / 36.0 + (-x0 + x1 + 2.0) * (-x0 + x1 + 2.0) / 8.0 - 17.0
        f3 = (x0 + 2 * x1 - 1) * (x0 + 2 * x1 - 1) / 175.0 + (2 * x1 - x0) * (2 * x1 - x0) / 17.0 - 13.0

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3

    def get_name(self):
        return "Viennet2"


class ZDT1(FloatProblem):
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
        return 1.0 - sqrt(f/g)

    def get_name(self):
        return "ZDT1"


class ZDT2(FloatProblem):
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
        return 1.0 - pow(f/g, 2.0)

    def get_name(self):
        return "ZDT2"


class ZDT3(FloatProblem):
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
        return 1.0 - sqrt(f/g) - (f/g)*sin(10.0*f*pi)

    def get_name(self):
        return "ZDT3"


class ZDT4(FloatProblem):
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

        for i in range(1, self.number_of_variables):
            g += pow(solution.variables[i], 2.0) - 10.0 * cos(4.0*pi*solution.variables[i])

        g += 1.0 + 10.0 * (solution.number_of_variables - 1)

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f/g)

    def get_name(self):
        return "ZDT4"


class ZDT6(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution):
        f0 = 1.0 - exp((-4.0) * solution.variables[0]) \
                   * pow(sin(6.0 * pi * solution.variables[0]), 6.0)
        g = self.__eval_g(solution)
        h = self.__eval_h(f0, g)

        solution.objectives[0] = f0
        solution.objectives[1] = h * g

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]
        g = g / (solution.number_of_variables - 1)
        g = pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f/g, 2.0)

    def get_name(self):
        return "ZDT6"
