from math import sqrt, exp, pow, sin

from jmetal.core.objective import Objective
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

""" Unconstrained Test problems for multi-objective optimization """

class Kursawe(FloatProblem):
    """ Class representing problem Kursawe """
    def __init__(self, number_of_variables: int = 3):
        self.objectives = [self.Objective1(), self.Objective2()]

        self.number_of_objectives = len(self.objectives)
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.lower_bound = [-5.0 for i in range(number_of_variables)]
        self.upper_bound = [5.0 for i in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self):
        return "Kursawe"

    class Objective1(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            fx = 0.0
            for i in range(problem.number_of_variables - 1):
                xi = solution.variables[i] * solution.variables[i]
                xj = solution.variables[i + 1] * solution.variables[i + 1]
                aux = -0.2 * sqrt(xi + xj)
                fx += -10 * exp(aux)

            return fx

    class Objective2(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            fx = 0.0
            for i in range(problem.number_of_variables):
                fx += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

            return fx


class Fonseca(FloatProblem):
    def __init__(self):
        self.objectives = [self.Objective1(), self.Objective2()]

        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [ 4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self):
        return "Fonseca"

    class Objective1(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            n = problem.number_of_variables

            return 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

    class Objective2(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            n = problem.number_of_variables

            return 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

class Schaffer(FloatProblem):
    def __init__(self):
        self.objectives = [self.Objective1(), self.Objective2()]

        self.number_of_variables = 1
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = [-100000]
        self.upper_bound = [100000]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self):
        return "Schaffer"

    class Objective1(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            return solution.variables[0] ** 2

    class Objective2(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            return (solution.variables[0] - 2.0) ** 2


class Viennet2(FloatProblem):
    def __init__(self):
        self.objectives = [self.Objective1(), self.Objective2(), self.Objective3()]

        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self):
        return "Viennet2"

    class Objective1(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            x0 = solution.variables[0]
            x1 = solution.variables[1]

            return (x0 - 2) * (x0 - 2) / 2.0 + (x1 + 1) * (x1 + 1) / 13.0 + 3.0

    class Objective2(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            x0 = solution.variables[0]
            x1 = solution.variables[1]

            return (x0 + x1 - 3.0) * (x0 + x1 - 3.0) / 36.0 + (-x0 + x1 + 2.0) * (-x0 + x1 + 2.0) / 8.0 - 17.0

    class Objective3(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            x0 = solution.variables[0]
            x1 = solution.variables[1]

            return (x0 + 2 * x1 - 1) * (x0 + 2 * x1 - 1) / 175.0 + (2 * x1 - x0) * (2 * x1 - x0) / 17.0 - 13.0
