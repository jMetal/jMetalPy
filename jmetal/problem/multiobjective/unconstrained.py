import random
from math import sqrt, exp, pow, sin

from jmetal.core.problem import FloatProblem, BinaryProblem, Problem
from jmetal.core.solution import FloatSolution, BinarySolution, CompositeSolution, IntegerSolution

"""
.. module:: constrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for multi-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Kursawe(FloatProblem):
    """ Class representing problem Kursawe. """

    def __init__(self, number_of_variables: int = 3):
        super(Kursawe, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-5.0 for _ in range(number_of_variables)]
        self.upper_bound = [5.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        fx = [0.0 for _ in range(self.number_of_objectives)]
        for i in range(self.number_of_variables - 1):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)
            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

        return solution

    def get_name(self):
        return 'Kursawe'


class Fonseca(FloatProblem):

    def __init__(self):
        super(Fonseca, self).__init__()
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        n = self.number_of_variables
        solution.objectives[0] = 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))
        solution.objectives[1] = 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

        return solution

    def get_name(self):
        return 'Fonseca'


class Schaffer(FloatProblem):

    def __init__(self):
        super(Schaffer, self).__init__()
        self.number_of_variables = 1
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = [-100000]
        self.upper_bound = [100000]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        value = solution.variables[0]

        solution.objectives[0] = value ** 2
        solution.objectives[1] = (value - 2) ** 2

        return solution

    def get_name(self):
        return 'Schaffer'


class Viennet2(FloatProblem):

    def __init__(self):
        super(Viennet2, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x0 = solution.variables[0]
        x1 = solution.variables[1]

        f1 = (x0 - 2) * (x0 - 2) / 2.0 + (x1 + 1) * (x1 + 1) / 13.0 + 3.0
        f2 = (x0 + x1 - 3.0) * (x0 + x1 - 3.0) / 36.0 + (-x0 + x1 + 2.0) * (-x0 + x1 + 2.0) / 8.0 - 17.0
        f3 = (x0 + 2 * x1 - 1) * (x0 + 2 * x1 - 1) / 175.0 + (2 * x1 - x0) * (2 * x1 - x0) / 17.0 - 13.0

        solution.objectives[0] = f1
        solution.objectives[1] = f2
        solution.objectives[2] = f3

        return solution

    def get_name(self):
        return 'Viennet2'


class SubsetSum(BinaryProblem):

    def __init__(self, C: int, W: list):
        """ The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.

        :param C: Large integer.
        :param W: Set of non-negative integers."""
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 2
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
        self.obj_labels = ['Sum', 'No. of Objects']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_sum = 0.0
        number_of_objects = 0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_sum += self.W[index]
                number_of_objects += 1

        if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

        solution.objectives[0] = -1.0 * total_sum
        solution.objectives[1] = number_of_objects

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'Subset Sum'


class OneZeroMax(BinaryProblem):

    def __init__(self, number_of_bits: int = 256):
        super(OneZeroMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 2
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        counter_of_ones = 0
        counter_of_zeroes = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1
            else:
                counter_of_zeroes += 1

        solution.objectives[0] = -1.0 * counter_of_ones
        solution.objectives[1] = -1.0 * counter_of_zeroes

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def get_name(self) -> str:
        return 'OneZeroMax'


class MixedIntegerFloatProblem(Problem):
    def __init__(self, number_of_integer_variables=10, number_of_float_variables=10, n=100, m=-100, lower_bound=-1000,
                 upper_bound=1000):
        super(MixedIntegerFloatProblem, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = 2
        self.number_of_constraints = 0

        self.n = n
        self.m = m

        self.float_lower_bound = [lower_bound for _ in range(number_of_float_variables)]
        self.float_upper_bound = [upper_bound for _ in range(number_of_float_variables)]
        self.int_lower_bound = [lower_bound for _ in range(number_of_integer_variables)]
        self.int_upper_bound = [upper_bound for _ in range(number_of_integer_variables)]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: CompositeSolution) -> CompositeSolution:
        distance_to_n = sum([abs(self.n - value) for value in solution.variables[0].variables])
        distance_to_m = sum([abs(self.m - value) for value in solution.variables[0].variables])

        distance_to_n += sum([abs(self.n - value) for value in solution.variables[1].variables])
        distance_to_m += sum([abs(self.m - value) for value in solution.variables[1].variables])

        solution.objectives[0] = distance_to_n
        solution.objectives[1] = distance_to_m

        return solution

    def create_solution(self) -> CompositeSolution:
        integer_solution = IntegerSolution(self.int_lower_bound, self.int_upper_bound, self.number_of_objectives,
                                           self.number_of_constraints)
        float_solution = FloatSolution(
            self.float_lower_bound,
            self.float_upper_bound,
            self.number_of_objectives, self.number_of_constraints)

        float_solution.variables = \
            [random.uniform(self.float_lower_bound[i] * 1.0, self.float_upper_bound[i] * .01) for i in
             range(len(self.int_lower_bound))]

        integer_solution.variables = \
            [random.uniform(self.float_lower_bound[i], self.float_upper_bound[i]) for i in
             range(len(self.float_lower_bound))]

        return CompositeSolution([integer_solution, float_solution])

    def get_name(self) -> str:
        return "Mixed Integer Float Problem"

