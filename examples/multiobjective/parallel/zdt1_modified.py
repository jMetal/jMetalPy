from math import sqrt

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class ZDT1Modified(FloatProblem):
    """ Problem ZDT1.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """

    def __init__(self, number_of_variables: int=30):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(ZDT1Modified, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        s: float = 0.0
        for i in range(1000):
            for j in range(10000):
                s += i * 0.235 / 1.234 + 1.23525 * j

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)
        g = constant * g
        g = g + 1.0

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'ZDT11'
