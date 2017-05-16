from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution.floatSolution import FloatSolution

""" Class representing Schaffer problem"""

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
