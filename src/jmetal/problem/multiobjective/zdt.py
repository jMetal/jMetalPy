import random
from math import cos, pi, pow, sin, sqrt, exp

from jmetal.core.problem import FloatProblem, BinaryProblem
from jmetal.core.solution import FloatSolution, BinarySolution

"""
.. module:: ZDT
   :platform: Unix, Windows
   :synopsis: ZDT problem family of multi-objective problems.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ZDT1(FloatProblem):
    """Problem ZDT1.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int = 30):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT1, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = number_of_variables * [0.0]
        self.upper_bound = number_of_variables * [1.0]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.eval_g(solution)
        h = self.eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (len(solution.variables) - 1)

        return constant * g + 1.0

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def name(self):
        return "ZDT1"


class ZDT1Modified(ZDT1):
    """Problem ZDT1Modified.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """

    def __init__(self, number_of_variables=30):
        super(ZDT1Modified, self).__init__(number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        s: float = 0.0
        for i in range(1000):
            for j in range(10000):
                s += i * 0.235 / 1.234 + 1.23525 * j
        return super().evaluate(solution)


class ZDT1Modified(ZDT1):
    """ Problem ZDT1Modified.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """
    def __init__(self, number_of_variables = 30):
        super(ZDT1Modified, self).__init__(number_of_variables)

    def evaluate(self, solution:FloatSolution) -> FloatSolution:
        s: float = 0.0
        for i in range(1000):
            for j in range(10000):
                s += i * 0.235 / 1.234 + 1.23525 * j
        return super().evaluate(solution)


class ZDT2(ZDT1):
    """Problem ZDT2.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def name(self):
        return "ZDT2"


class ZDT3(ZDT1):
    """Problem ZDT3.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
    .. note:: Continuous problem having a partitioned Pareto front
    """

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g) - (f / g) * sin(10.0 * f * pi)

    def name(self):
        return "ZDT3"


class ZDT4(ZDT1):
    """Problem ZDT4.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous multi-modal problem having a convex Pareto front
    """

    def __init__(self, number_of_variables: int = 10):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT4, self).__init__()
        self.lower_bound = number_of_variables * [-5.0]
        self.upper_bound = number_of_variables * [5.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def eval_g(self, solution: FloatSolution):
        g = 0.0

        for i in range(1, len(solution.variables)):
            g += pow(solution.variables[i], 2.0) - 10.0 * cos(4.0 * pi * solution.variables[i])

        g += 1.0 + 10.0 * (len(solution.variables) - 1)

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def name(self):
        return "ZDT4"


class ZDT5(BinaryProblem):
    """Problem ZDT5.

    .. note:: Bi-objective binary unconstrained problem. The default number of variables is 11.
    
    In this implementation, each variable is represented by a single boolean value in the solution,
    and the number_of_bits_per_variable attribute is used to track how many bits each variable
    conceptually represents for evaluation purposes.
    """

    def __init__(self, number_of_variables: int = 11):
        """
        :param number_of_variables: Number of variables in the problem.
        """
        super(ZDT5, self).__init__()

        # Track how many bits each variable conceptually represents
        self.number_of_bits_per_variable = [5 for _ in range(number_of_variables)]
        self.number_of_bits_per_variable[0] = 30
        
        # Total number of bits is the sum of all bits per variable
        self.total_number_of_bits = sum(self.number_of_bits_per_variable)
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]
        
        # For compatibility with the original implementation
        self.number_of_bits = self.total_number_of_bits

    def number_of_variables(self) -> int:
        return self.total_number_of_bits

    def total_number_of_bits(self) -> int:
        return self.total_number_of_bits

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        """
        Evaluate the solution by counting the number of true bits in each variable's range.
        """
        # Calculate first objective: 1 + number of true bits in first variable (30 bits)
        first_var_bits = solution.variables[:30]
        solution.objectives[0] = 1.0 + sum(first_var_bits)

        # Calculate g function for second objective
        g = self.eval_g(solution)
        h = 1.0 / solution.objectives[0]
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: BinarySolution) -> float:
        """
        Calculate the g function for ZDT5.
        """
        result = 0.0
        bit_index = 30  # Start after the first variable (30 bits)
        
        # Process remaining variables (each 5 bits)
        for bits in self.number_of_bits_per_variable[1:]:
            # Count true bits in this variable's range
            var_bits = solution.variables[bit_index:bit_index + bits]
            ones_count = sum(var_bits)
            result += self.eval_v(ones_count)
            bit_index += bits
            
        return result

    def eval_v(self, value: int) -> float:
        """
        Helper function for ZDT5 evaluation.
        """
        if value < 5.0:
            return 2.0 + value
        return 1.0

    def create_solution(self) -> BinarySolution:
        """
        Create a new random solution.
        """
        solution = BinarySolution(
            number_of_variables=self.total_number_of_bits,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints()
        )
        
        # Initialize with random bits
        for i in range(self.total_number_of_bits):
            solution.variables[i] = random.random() < 0.5
            
        return solution

    def name(self) -> str:
        return "ZDT5"


class ZDT6(ZDT1):
    """Problem ZDT6.

    .. note:: Bi-objective unconstrained problem. The default number of variables is 10.
    .. note:: Continuous problem having a non-convex Pareto front
    """

    def __init__(self, number_of_variables: int = 10):
        """:param number_of_variables: Number of decision variables of the problem."""
        super(ZDT6, self).__init__(number_of_variables=number_of_variables)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = (
            1.0 - exp(-4.0 * solution.variables[0]) * (sin(6.0 * pi * solution.variables[0])) ** 6.0
        )

        g = self.eval_g(solution)
        h = self.eval_h(solution.objectives[0], g)
        solution.objectives[1] = h * g

        return solution

    def eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]
        g = g / (len(solution.variables) - 1)
        g = pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g

        return g

    def eval_h(self, f: float, g: float) -> float:
        return 1.0 - pow(f / g, 2.0)

    def name(self):
        return "ZDT6"
