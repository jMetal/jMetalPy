from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution

""" Class representing Viennet2 problem"""

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
