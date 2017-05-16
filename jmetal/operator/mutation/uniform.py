import random

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.floatSolution import FloatSolution

""" Class implementing the float uniform mutation operator """


class Uniform(MutationOperator[FloatSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5):
        super(Uniform, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                tmp = (random.random() - 0.5) * self.perturbation;
                tmp+= solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

