import random
from jmetal.core.operator import MutationOperator
from jmetal.core.solution import BinarySolution

<<<<<<< HEAD:jmetal/operator/mutation/bitflip.py
from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.solution.binarySolution import BinarySolution

""" Class implementing the binary BitFlip mutation operator """
__author__ = "Antonio J. Nebro"


class BitFlip(MutationOperator[BinarySolution]):
    def __init__(self, probability: float):
        super(BitFlip, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    solution.variables[i][j] = True if solution.variables[i][j] == False else False

        return solution

=======
__author__ = "Antonio J. Nebro"


class BitFlip(MutationOperator):
    """ Class implementing the binary BitFlip mutation operator """

    def __init__(self, number_of_bits: int = 256):
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

    def evaluate(self, solution: BinarySolution) -> None:
        counter_of_ones = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1
        solution.objectives[0] = counter_of_ones

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for i in range(self.number_of_bits)]
        return new_solution
>>>>>>> 0c3a3b5ecb116c4ec22fd8540d233f554fdd700a:jmetal/operator/mutation.py
