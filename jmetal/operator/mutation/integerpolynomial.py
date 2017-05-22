import random

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.integerSolution import IntegerSolution

""" Class implementing the integer polynomial mutation operator """

class IntegerPolynomial(MutationOperator[IntegerSolution]):
    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(IntegerPolynomial, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                y = solution.variables[i]
                yl = solution.lower_bound[i]
                yu = solution.upper_bound[i]
                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mutPow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd<=0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val**mutPow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy**(self.distribution_index + 1.0))
                        deltaq = 1.0 - val**mutPow

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = int(round(y))
        return solution
