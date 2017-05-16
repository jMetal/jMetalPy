import random

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.floatSolution import FloatSolution

""" Class implementing the float simple mutation operator """


class SimpleRandom(MutationOperator[FloatSolution]):
    def __init__(self, probability: float):
        super(SimpleRandom, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = solution.lower_bound[i] + (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
        return solution

