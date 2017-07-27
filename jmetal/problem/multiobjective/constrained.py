

""" Unconstrained Test problems for multi-objective optimization """
from jmetal.core.objective import Objective
from jmetal.core.solution import FloatSolution

from jmetal.core.problem import FloatProblem


class Srinivas(FloatProblem):
    """ Class representing problem Kursawe """
    def __init__(self):
        self.objectives = [self.Objective1(), self.Objective2()]

        self.number_of_objectives = len(self.objectives)
        self.number_of_variables = 2
        self.number_of_constraints = 2

        self.lower_bound = [-20.0 for i in range(self.number_of_variables)]
        self.upper_bound = [20.0 for i in range(self.number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self):
        return "Srinivas"

    class Objective1(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            x1 = solution.variables[0]
            x2 = solution.variables[1]

            return 2.0 + (x1 - 2.0) * (x1 - 2.0) + (x2 - 1.0) * (x2 - 1.0)

    class Objective2(Objective):
        def compute(self, solution: FloatSolution, problem: FloatProblem):
            x1 = solution.variables[0]
            x2 = solution.variables[1]

            return 9.0 * x1 - (x2 - 1.0) * (x2 - 1.0)

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints : [float] = [self.number_of_constraints]

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

        solution.attributes["overall_constraint_violation"] = overall_constraint_violation
        solution.attributes["number_of_violated_constraints"] = number_of_violated_constraints