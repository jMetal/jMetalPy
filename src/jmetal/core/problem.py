"""
This module defines the core problem interfaces for optimization in JMetalPy.

It provides abstract base classes for defining optimization problems of various types,
including binary, float, integer, and permutation problems, as well as utilities for
creating problems on the fly.
"""

import random
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from jmetal.core.observer import Observer
from jmetal.core.solution import (
    BinarySolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
)
from jmetal.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Type variable for solution types
S = TypeVar("S")  # Generic type for solutions


class Problem(Generic[S], ABC):
    """Abstract base class for all optimization problems.
    
    This class defines the common interface that all optimization problems must implement.
    It serves as the foundation for defining problems with different variable types
    (binary, float, integer, permutation) and characteristics (single/multi-objective,
    constrained/unconstrained).
    
    Class constants:
        MINIMIZE: Constant indicating minimization of an objective.
        MAXIMIZE: Constant indicating maximization of an objective.
    
    Attributes:
        reference_front: List of solutions representing the Pareto front (for multi-objective problems).
        directions: List indicating optimization direction (minimize/maximize) for each objective.
        labels: List of descriptive labels for each objective.
    """

    # Optimization directions
    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self):
        """Initialize the problem with empty reference front, directions, and labels."""
        self.reference_front: List[S] = []
        self.directions: List[int] = []
        self.labels: List[str] = []

    @abstractmethod
    def number_of_variables(self) -> int:
        pass

    @abstractmethod
    def number_of_objectives(self) -> int:
        pass

    @abstractmethod
    def number_of_constraints(self) -> int:
        pass

    @abstractmethod
    def create_solution(self) -> S:
        """Creates a random_search solution to the problem.

        :return: Solution."""
        pass

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be replaced.
        Note that this framework ASSUMES minimization, thus solutions must be evaluated in consequence.

        :return: Evaluated solution."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class DynamicProblem(Problem[S], Observer, ABC):
    """Abstract base class for dynamic optimization problems.
    
    Dynamic problems are those where the fitness landscape, constraints, or other
    characteristics may change over time. This class extends the base Problem
    interface with methods to detect and handle such changes.
    
    This class also implements the Observer pattern to allow the problem to be
    notified of changes in the environment or other components.
    
    The type parameter S represents the type of the solution this problem works with.
    """
    
    @abstractmethod
    def the_problem_has_changed(self) -> bool:
        """Check if the problem has changed since the last check.
        
        Returns:
            bool: True if the problem has changed, False otherwise.
        """
        pass

    @abstractmethod
    def clear_changed(self) -> None:
        """Clear the changed flag after handling a change event.
        
        This method should be called after the algorithm has responded to a change
        in the problem to reset the change detection mechanism.
        """
        pass


class BinaryProblem(Problem[BinarySolution], ABC):
    """Abstract base class for binary-encoded optimization problems.
    
    This class is designed for problems where solutions are represented as bit strings.
    Each variable in the problem is encoded using a fixed number of bits, which can
    vary between variables.
    
    Attributes:
        number_of_bits_per_variable: List specifying the number of bits used to encode each variable.
    """

    def __init__(self):
        """Initialize a binary problem with an empty list of bits per variable."""
        super(BinaryProblem, self).__init__()
        self.number_of_bits_per_variable = []

    def number_of_bits_per_variable_list(self):
        return self.number_of_bits_per_variable

    def total_number_of_bits(self):
        return sum(self.number_of_bits_per_variable)


class FloatProblem(Problem[FloatSolution], ABC):
    """Abstract base class for continuous optimization problems with float variables.
    
    This class is designed for problems where decision variables can take any real
    value within specified lower and upper bounds. It's suitable for continuous
    optimization problems in any number of dimensions.
    
    Attributes:
        lower_bound: List of lower bounds for each decision variable.
        upper_bound: List of upper bounds for each decision variable.
    """

    def __init__(self):
        """Initialize a float problem with empty bounds."""
        super(FloatProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives(), self.number_of_constraints()
        )
        new_solution.variables = [
            random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
            for i in range(self.number_of_variables())
        ]

        return new_solution


class IntegerProblem(Problem[IntegerSolution], ABC):
    """Abstract base class for integer-constrained optimization problems.
    
    This class is designed for problems where decision variables must take integer
    values within specified lower and upper bounds. It's suitable for discrete
    optimization problems, combinatorial problems, and mixed-integer problems.
    
    Attributes:
        lower_bound: List of lower bounds (inclusive) for each decision variable.
        upper_bound: List of upper bounds (inclusive) for each decision variable.
    """

    def __init__(self):
        """Initialize an integer problem with empty bounds."""
        super(IntegerProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives(), self.number_of_constraints()
        )
        new_solution.variables = [
            round(random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0))
            for i in range(self.number_of_variables())
        ]

        return new_solution


class PermutationProblem(Problem[PermutationSolution], ABC):
    """Abstract base class for permutation-based optimization problems.
    
    This class is designed for problems where solutions are represented as permutations
    of a set of elements. Common applications include routing problems (like TSP),
    scheduling problems, and other combinatorial optimization problems where the order
    of elements is significant.
    
    The permutation is represented as a list of integers from 0 to n-1, where n is
    the number of elements in the permutation.
    """

    def __init__(self):
        """Initialize a permutation problem."""
        super(PermutationProblem, self).__init__()


class OnTheFlyFloatProblem(FloatProblem):
    """A utility class for defining float optimization problems dynamically at runtime.
    
    This class allows users to define optimization problems programmatically by
    specifying the problem's variables, objectives, and constraints through method
    chaining. It's particularly useful for quick prototyping and testing.
    
    Example:
        # Define the problem's objective functions and constraints
        def f1(x: List[float]) -> float:
            return 2.0 + (x[0] - 2.0)**2 + (x[1] - 1.0)**2
            
        def f2(x: List[float]) -> float:
            return 9.0 * x[0] - (x[1] - 1.0)**2
            
        def c1(x: List[float]) -> float:
            return 1.0 - (x[0]**2 + x[1]**2) / 225.0
            
        def c2(x: List[float]) -> float:
            return (3.0 * x[1] - x[0]) / 10.0 - 1.0
        
        # Create the problem with method chaining
        problem = (OnTheFlyFloatProblem()
                  .set_name("Srinivas")
                  .add_variable(-20.0, 20.0)  # x1 ∈ [-20, 20]
                  .add_variable(-20.0, 20.0)  # x2 ∈ [-20, 20]
                  .add_function(f1)            # First objective
                  .add_function(f2)            # Second objective
                  .add_constraint(c1)          # First constraint (g1(x) ≤ 0)
                  .add_constraint(c2))         # Second constraint (g2(x) ≤ 0)
    
    Attributes:
        functions: List of objective functions to be minimized.
        constraints: List of constraint functions (≤ 0).
        problem_name: Optional name for the problem.
    """

    def __init__(self):
        super(OnTheFlyFloatProblem, self).__init__()
        self.functions = []
        self.constraints = []
        self.problem_name = None

    def set_name(self, name) -> "OnTheFlyFloatProblem":
        self.problem_name = name

        return self

    def add_function(self, function) -> "OnTheFlyFloatProblem":
        self.functions.append(function)

        return self

    def add_constraint(self, constraint) -> "OnTheFlyFloatProblem":
        self.constraints.append(constraint)

        return self

    def add_variable(self, lower_bound, upper_bound) -> "OnTheFlyFloatProblem":
        self.lower_bound.append(lower_bound)
        self.upper_bound.append(upper_bound)

        return self

    def number_of_objectives(self) -> int:
        return len(self.functions)

    def number_of_constraints(self) -> int:
        return len(self.constraints)

    def evaluate(self, solution: FloatSolution) -> None:
        for i in range(self.number_of_objectives()):
            solution.objectives[i] = self.functions[i](solution.variables)

        for i in range(self.number_of_constraints()):
            solution.constraints[i] = self.constraints[i](solution.variables)

    def name(self) -> str:
        return self.problem_name
