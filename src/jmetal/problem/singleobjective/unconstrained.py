import math
import random
import numpy as np

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution

"""
.. module:: unconstrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for single-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class OneMax(BinaryProblem):
    """The OneMax problem is a simple optimization problem that counts the number of ones in a binary string.
    
    The objective is to maximize the number of ones in the binary string, which is equivalent to
    minimizing the negative count of ones.
    
    Args:
        number_of_bits: The length of the binary string (default: 256)
    """
    def __init__(self, number_of_bits: int = 256):
        super(OneMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_bits_per_variable = [number_of_bits]  # For backward compatibility

        self.obj_directions = [self.MINIMIZE]  # We'll use negative count for minimization
        self.obj_labels = ["Ones"]

    def number_of_variables(self) -> int:
        return self.number_of_bits  # Each bit is treated as a variable

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # Count the number of ones in the binary string
        counter_of_ones = np.count_nonzero(solution.bits)
        
        # Store the negative count to be minimized (equivalent to maximizing the positive count)
        solution.objectives[0] = -float(counter_of_ones)
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
        return "OneMax"


class Sphere(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Sphere, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

        return solution

    def name(self) -> str:
        return "Sphere"


class Rastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0
        result = a * len(solution.variables)
        x = solution.variables

        for i in range(len(solution.variables)):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])

        solution.objectives[0] = result

        return solution

    def name(self) -> str:
        return "Rastrigin"


class SubsetSum(BinaryProblem):
    def __init__(self, C: int, W: list):
        """The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.
        
        This is a single-objective problem where we want to:
        1. Maximize the sum of selected elements (without exceeding C)
        
        Args:
            C: The target sum (large integer)
            W: List of non-negative integers to select from
        """
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = np.array(W, dtype=float)  # Convert to numpy array for vectorized operations
        
        self.number_of_bits = len(self.W)
        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ["Sum"]

    def number_of_variables(self) -> int:
        return self.number_of_bits  # Each bit represents whether an item is selected

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # Get the mask of selected items (bits that are True)
        selected_mask = solution.bits
        
        # Calculate total sum of selected items
        total_sum = np.sum(self.W[selected_mask])
        
        # Penalize solutions that exceed the target sum C
        if total_sum > self.C:
            # Apply a penalty that increases with how much we exceed C
            total_sum = self.C - (total_sum - self.C)
            if total_sum < 0.0:
                total_sum = 0.0
        
        # Store the negative sum to be maximized
        solution.objectives[0] = -total_sum
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
