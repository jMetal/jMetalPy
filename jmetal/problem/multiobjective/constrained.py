from math import pi, cos, atan

from jmetal.core.solution import FloatSolution
from jmetal.core.problem import FloatProblem

"""
.. module:: constrained
   :platform: Unix, Windows
   :synopsis: Constrained test problems for multi-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Srinivas(FloatProblem):
    """ Class representing problem Srinivas. """

    def __init__(self, rf_path: str=None):
        super(Srinivas, self).__init__(rf_path=rf_path)
        self.number_of_objectives = 2
        self.number_of_variables = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-20.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [20.0 for _ in range(self.number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        solution.objectives[0] = 2.0 + (x1 - 2.0) * (x1 - 2.0) + (x2 - 1.0) * (x2 - 1.0)
        solution.objectives[1] = 9.0 * x1 - (x2 - 1.0) * (x2 - 1.0)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        x1 = solution.variables[0]
        x2 = solution.variables[1]

        constraints[0] = 1.0 - (x1 * x1 + x2 * x2) / 225.0
        constraints[1] = (3.0 * x2 - x1) / 10.0 - 1.0

        overall_constraint_violation = 0.0
        number_of_violated_constraints = 0.0

        for constrain in constraints:
            if constrain < 0.0:
                overall_constraint_violation += constrain
                number_of_violated_constraints += 1

        solution.attributes['overall_constraint_violation'] = overall_constraint_violation
        solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

    def get_name(self):
        return 'Srinivas'


class Tanaka(FloatProblem):
    """ Class representing problem Tanaka """

    def __init__(self, rf_path: str=None):
        super(Tanaka, self).__init__(rf_path=rf_path)
        self.number_of_objectives = 2
        self.number_of_variables = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [10e-5 for _ in range(self.number_of_variables)]
        self.upper_bound = [pi for _ in range(self.number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = solution.variables[1]

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        x1 = solution.variables[0]
        x2 = solution.variables[1]

        constraints[0] = (x1 * x1 + x2 * x2 - 1.0 - 0.1 * cos(16.0 * atan(x1 / x2)))
        constraints[1] = -2.0 * ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) - 0.5)

        overall_constraint_violation = 0.0
        number_of_violated_constraints = 0.0

        for constrain in constraints:
            if constrain < 0.0:
                overall_constraint_violation += constrain
                number_of_violated_constraints += 1

        solution.attributes['overall_constraint_violation'] = overall_constraint_violation
        solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

    def get_name(self):
        return 'Tanaka'
