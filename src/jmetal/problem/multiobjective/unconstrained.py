import random
from math import exp, pow, sin, sqrt
import numpy as np

from jmetal.core.problem import BinaryProblem, FloatProblem, Problem
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
)

"""
.. module:: constrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for multi-objective optimization

.. moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


class Kursawe(FloatProblem):
    """Class representing problem Kursawe."""

    def __init__(self, number_of_variables: int = 3):
        super(Kursawe, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [-5.0 for _ in range(number_of_variables)]
        self.upper_bound = [5.0 for _ in range(number_of_variables)]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        fx = [0.0 for _ in range(self.number_of_objectives())]
        for i in range(self.number_of_variables() - 1):
            xi = solution.variables[i] * solution.variables[i]
            xj = solution.variables[i + 1] * solution.variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)

        for i in range(self.number_of_variables()):
            fx[1] += pow(abs(solution.variables[i]), 0.8) + 5.0 * sin(pow(solution.variables[i], 3.0))

        solution.objectives[0] = fx[0]
        solution.objectives[1] = fx[1]

        return solution

    def name(self):
        return "Kursawe"


class Fonseca(FloatProblem):
    def __init__(self):
        super(Fonseca, self).__init__()
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        number_of_variables = 3

        self.lower_bound = number_of_variables * [-4]
        self.upper_bound = number_of_variables * [4]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        n = self.number_of_variables()
        solution.objectives[0] = 1 - exp(-sum([(x - 1.0 / n ** 0.5) ** 2 for x in solution.variables]))
        solution.objectives[1] = 1 - exp(-sum([(x + 1.0 / n ** 0.5) ** 2 for x in solution.variables]))

        return solution

    def name(self):
        return "Fonseca"


class Schaffer(FloatProblem):
    def __init__(self):
        super(Schaffer, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)"]

        self.lower_bound = [-1000]
        self.upper_bound = [1000]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        value = solution.variables[0]

        solution.objectives[0] = value ** 2
        solution.objectives[1] = (value - 2) ** 2

        return solution

    def name(self):
        return "Schaffer"


class Viennet2(FloatProblem):
    def __init__(self):
        super(Viennet2, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["f(x)", "f(y)", "f(z)"]

        number_of_variables = 2
        self.lower_bound = number_of_variables * [-4]
        self.upper_bound = number_of_variables * [4]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

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

    def name(self):
        return "Viennet2"


class SubsetSum(BinaryProblem):
    def __init__(self, C: int, W: list):
        """The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.
        
        This is a bi-objective problem where we want to:
        1. Maximize the sum of selected elements (without exceeding C)
        2. Minimize the number of selected objects
        
        Args:
            C: The target sum (large integer)
            W: List of non-negative integers to select from
        """
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = np.array(W, dtype=float)  # Convert to numpy array for vectorized operations
        
        self.number_of_bits = len(self.W)
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        # Objective 1: Maximize sum (minimize negative sum)
        # Objective 2: Minimize number of selected objects
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
        self.obj_labels = ["Sum", "No. of Objects"]

    def number_of_variables(self) -> int:
        return self.number_of_bits  # Each bit represents whether an item is selected

    def number_of_objectives(self) -> int:
        return self.number_of_objectives

    def number_of_constraints(self) -> int:
        return self.number_of_constraints

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # Get the mask of selected items (bits that are True)
        selected_mask = solution.bits
        
        # Calculate total sum of selected items
        total_sum = np.sum(self.W[selected_mask])
        number_of_objects = np.count_nonzero(selected_mask)
        
        # Penalize solutions that exceed the target sum C
        if total_sum > self.C:
            total_sum = self.C - (total_sum - self.C)  # Penalize by how much it exceeds
            if total_sum < 0.0:
                total_sum = 0.0
        
        # Store objectives
        # Note: First objective is negated because we're using MAXIMIZE direction
        solution.objectives[0] = -total_sum  # Will be maximized
        solution.objectives[1] = number_of_objects  # To be minimized
        
        return solution

    def create_solution(self) -> BinarySolution:
        # Create a new binary solution with one bit per item in W
        solution = BinarySolution(
            number_of_variables=self.number_of_bits,
            number_of_objectives=self.number_of_objectives()
        )
        
        # Initialize with random bits (each bit represents whether an item is selected)
        solution.bits = np.random.choice([True, False], size=self.number_of_bits)
        
        return solution

    def name(self) -> str:
        return "Subset Sum"


class OneZeroMax(BinaryProblem):
    """The OneZeroMax problem is a multi-objective problem that counts the number of ones and zeros in a binary string.
    
    The objectives are:
    1. Maximize the number of ones (minimize negative count)
    2. Maximize the number of zeros (minimize negative count)
    
    Args:
        number_of_bits: The length of the binary string (default: 256)
    """

    def __init__(self, number_of_bits: int = 256):
        super(OneZeroMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_bits_per_variable = [number_of_bits]  # For backward compatibility

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Ones", "Zeros"]

    def number_of_variables(self) -> int:
        return self.number_of_bits  # Each bit is treated as a variable

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # Count the number of ones and zeros in the binary string
        counter_of_ones = np.count_nonzero(solution.bits)
        counter_of_zeros = len(solution.bits) - counter_of_ones

        # Store the negative counts to be minimized
        solution.objectives[0] = -1.0 * counter_of_ones
        solution.objectives[1] = -1.0 * counter_of_zeros

        return solution

    def create_solution(self) -> BinarySolution:
        # Create a new binary solution with the specified number of bits
        solution = BinarySolution(
            number_of_variables=self.number_of_bits,
            number_of_objectives=self.number_of_objectives()
        )
        
        # Initialize with random bits (using numpy for better performance)
        solution.bits = np.random.choice([True, False], size=self.number_of_bits)
        
        return solution

    def name(self) -> str:
        return "OneZeroMax"


class MixedIntegerFloatProblem(Problem):
    def __init__(
            self,
            number_of_integer_variables=10,
            number_of_float_variables=10,
            n=100,
            m=-100,
            lower_bound=-1000,
            upper_bound=1000,
    ):
        super(MixedIntegerFloatProblem, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = number_of_float_variables + number_of_integer_variables
        self.number_of_constraints = 0

        self.n = n
        self.m = m

        self.float_lower_bound = [lower_bound for _ in range(number_of_float_variables)]
        self.float_upper_bound = [upper_bound for _ in range(number_of_float_variables)]
        self.int_lower_bound = [lower_bound for _ in range(number_of_integer_variables)]
        self.int_upper_bound = [upper_bound for _ in range(number_of_integer_variables)]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Ones"]

    def number_of_constraints(self) -> int:
        return self.number_of_constraints

    def number_of_objectives(self) -> int:
        return self.number_of_objectives

    def number_of_variables(self) -> int:
        return self.number_of_variables

    def evaluate(self, solution: CompositeSolution) -> CompositeSolution:
        distance_to_n = sum([abs(self.n - value) for value in solution.variables[0].variables])
        distance_to_m = sum([abs(self.m - value) for value in solution.variables[0].variables])

        distance_to_n += sum([abs(self.n - value) for value in solution.variables[1].variables])
        distance_to_m += sum([abs(self.m - value) for value in solution.variables[1].variables])

        solution.objectives[0] = distance_to_n
        solution.objectives[1] = distance_to_m

        return solution

    def create_solution(self) -> CompositeSolution:
        integer_solution = IntegerSolution(
            self.int_lower_bound, self.int_upper_bound, self.number_of_objectives, self.number_of_constraints
        )
        float_solution = FloatSolution(
            self.float_lower_bound, self.float_upper_bound, self.number_of_objectives, self.number_of_constraints
        )

        float_solution.variables = [
            random.uniform(self.float_lower_bound[i] * 1.0, self.float_upper_bound[i] * 1.0)
            for i in range(len(self.float_lower_bound))
        ]
        integer_solution.variables = [
            random.uniform(self.int_lower_bound[i], self.int_upper_bound[i])
            for i in range(len(self.int_lower_bound))
        ]

        return CompositeSolution([integer_solution, float_solution])

    def name(self) -> str:
        return "Mixed Integer Float Problem"
