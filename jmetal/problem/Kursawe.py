from jmetal.core.problem.floatProblem import FloatProblem
from jmetal.core.solution import floatSolution

""" Class representing problem Kursawe """
__author__ = "Antonio J. Nebro"


class Kursawe(FloatProblem):
    def __init__(self, number_of_variables: int = 3):
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        upper_bound = [5.0 for i in range(number_of_variables)]
        lower_bound = [-5.0 for i in range(number_of_variables)]

        self.set_lower_bounds(lower_bound)
        self.set_upper_bounds(upper_bound)



    def evaluate(self, solution: floatSolution):
        '''

        :param solution:
        :return:
        '''
        '''

            fx[0] = 0.0;
    for (int var = 0; var < solution.getNumberOfVariables() - 1; var++) {
      xi = x[var] * x[var];
      xj = x[var + 1] * x[var + 1];
      aux = (-0.2) * Math.sqrt(xi + xj);
      fx[0] += (-10.0) * Math.exp(aux);
    }

    fx[1] = 0.0;

    for (int var = 0; var < solution.getNumberOfVariables(); var++) {
      fx[1] += Math.pow(Math.abs(x[var]), 0.8) +
        5.0 * Math.sin(Math.pow(x[var], 3.0));
    }

    solution.setObjective(0, fx[0]);
    solution.setObjective(1, fx[1]);
        '''