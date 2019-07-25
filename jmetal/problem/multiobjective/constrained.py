from math import pi, cos, atan

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

"""
.. module:: constrained
   :platform: Unix, Windows
   :synopsis: Constrained test problems for multi-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Srinivas(FloatProblem):
    """ Class representing problem Srinivas. """

    def __init__(self):
        super(Srinivas, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-20.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [20.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        solution.objectives[0] = 2.0 + (x1 - 2.0) * (x1 - 2.0) + (x2 - 1.0) * (x2 - 1.0)
        solution.objectives[1] = 9.0 * x1 - (x2 - 1.0) * (x2 - 1.0)

        self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        solution.constraints[0] = 1.0 - (x1 * x1 + x2 * x2) / 225.0
        solution.constraints[1] = (3.0 * x2 - x1) / 10.0 - 1.0

    def get_name(self):
        return 'Srinivas'


class Tanaka(FloatProblem):
    """ Class representing problem Tanaka. """

    def __init__(self):
        super(Tanaka, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [10e-5 for _ in range(self.number_of_variables)]
        self.upper_bound = [pi for _ in range(self.number_of_variables)]


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = solution.variables[1]

        self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        x1 = solution.variables[0]
        x2 = solution.variables[1]

        constraints[0] = (x1 * x1 + x2 * x2 - 1.0 - 0.1 * cos(16.0 * atan(x1 / x2)))
        constraints[1] = -2.0 * ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) - 0.5)

        solution.constraints = constraints

        #set_overall_constraint_violation_degree(solution)


    def get_name(self):
        return 'Tanaka'


class Osyczka2(FloatProblem):
    """ Class representing problem Osyczka2. """

    def __init__(self):
        super(Osyczka2, self).__init__()
        self.number_of_variables = 6
        self.number_of_objectives = 2
        self.number_of_constraints = 6

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        self.upper_bound = [10.0, 10.0, 5.0, 6.0, 5.0, 10.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        solution.objectives[0] = - (25.0 *
                                    (x[0] - 2.0) ** 2 +
                                    (x[1] - 2.0) ** 2 +
                                    (x[2] - 1.0) ** 2 +
                                    (x[3] - 4.0) ** 2 +
                                    (x[4] - 1.0) ** 2)

        solution.objectives[1] = sum([x[i] ** 2 for i in range(len(x))])

        self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        x = solution.variables
        constraints[0] = (x[0] + x[1]) / 2.0 - 1.0
        constraints[1] = (6.0 - x[0] - x[1]) / 6.0
        constraints[2] = (2.0 - x[1] + x[0]) / 2.0
        constraints[3] = (2.0 - x[0] + 3.0 * x[1]) / 2.0
        constraints[4] = (4.0 - (x[2] - 3.0) * (x[2] - 3.0) - x[3]) / 4.0
        constraints[5] = ((x[4] - 3.0) * (x[4] - 3.0) + x[5] - 4.0) / 4.0

        solution.constraints = constraints

    def get_name(self):
        return 'Osyczka2'


class Binh2(FloatProblem):
    """ Class representing problem Binh2. """

    def __init__(self):
        super(Binh2, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 2
        self.number_of_constraints = 2

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [0.0, 0.0]
        self.upper_bound = [5.0, 3.0]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        solution.objectives[0] = 4.0 * x[0] * x[0] + 4 * x[1] * x[1]
        solution.objectives[1] = (x[0] - 5.0) * (x[0] - 5.0) + (x[1] - 5.0) * (x[1] - 5.0)

        self.__evaluate_constraints(solution)

        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]

        x = solution.variables
        constraints[0] = -1.0 * (x[0] - 5) * (x[0] - 5) - x[1] * x[1] + 25.0
        constraints[1] = (x[0] - 8) * (x[0] - 8) + (x[1] + 3) * (x[1] + 3) - 7.7

    def get_name(self):
        return 'Binh2'
