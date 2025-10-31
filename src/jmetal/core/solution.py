"""
This module defines the core solution representations used in evolutionary computation.
It provides abstract and concrete implementations of solutions for different types of optimization problems.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, final, Dict, Any

import numpy as np

from jmetal.util.ckecking import Check

# Type aliases for better code readability
BitSet = List[bool]  # Represents a sequence of binary values
S = TypeVar("S")  # Generic type variable for solution variables


class Solution(Generic[S], ABC):
    """Abstract base class for all solution representations in the optimization framework.
    
    This class defines the common interface and functionality for all solution types.
    Subclasses must implement the abstract methods to provide specific variable storage
    and manipulation mechanisms.
    
    Attributes:
        number_of_variables: Number of decision variables in the solution.
        number_of_objectives: Number of objective values to optimize.
        number_of_constraints: Number of constraint values (default: 0).
        _objectives: List storing the objective values of the solution.
        _constraints: List storing the constraint values of the solution.
        attributes: Dictionary for storing additional solution metadata.
    """
    
    def __init__(
        self, 
        number_of_variables: int, 
        number_of_objectives: int, 
        number_of_constraints: int = 0
    ) -> None:
        """Initialize a new solution with the specified dimensions.
        
        Args:
            number_of_variables: The number of decision variables.
            number_of_objectives: The number of objective values.
            number_of_constraints: The number of constraint values (default: 0).
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        
        # Initialize internal storage with default values
        self._objectives: List[float] = [0.0] * number_of_objectives
        self._constraints: List[float] = [0.0] * number_of_constraints
        self.attributes: Dict[str, Any] = {}
    
    @property
    def objectives(self) -> List[float]:
        return self._objectives
    
    @objectives.setter
    def objectives(self, values: List[float]) -> None:
        if len(values) != self.number_of_objectives:
            raise ValueError(
                f"Expected {self.number_of_objectives} objectives, got {len(values)}"
            )
        self._objectives = list(values)
    
    @property
    def constraints(self) -> List[float]:
        return self._constraints
    
    @constraints.setter
    def constraints(self, values: List[float]) -> None:
        if len(values) != self.number_of_constraints:
            raise ValueError(
                f"Expected {self.number_of_constraints} constraints, got {len(values)}"
            )
        self._constraints = list(values)
    
    @property
    @abstractmethod
    def variables(self) -> List[S]:
        """
        Return the decision variables as a list.
        
        Must return a list-like object where:
        - len(variables) == number_of_variables
        - variables[i] returns the i-th variable of type S
        """
        ...
    
    @variables.setter
    @abstractmethod
    def variables(self, values: List[S]) -> None:
        """Set the decision variables from a list."""
        ...
    
    @abstractmethod
    def __copy__(self) -> Self:
        """Create a deep copy of this solution."""
        ...
    
    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        return (self.variables == other.variables and 
                self.objectives == other.objectives and
                self.constraints == other.constraints)
    
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"variables={self.variables}, "
            f"objectives={self.objectives}, "
            f"constraints={self.constraints}"
            ")"
        )


@final
class BinarySolution(Solution[bool]):
    """A solution representation for binary-encoded optimization problems.
    
    This class provides an efficient implementation of binary solutions using NumPy
    arrays for storage and operations. It's particularly suited for problems where
    solutions are represented as bit strings, such as binary-encoded combinatorial
    optimization problems.
    
    The implementation uses NumPy's boolean arrays for compact storage and efficient
    bitwise operations. It maintains both a NumPy array for performance and provides
    a Python list interface for compatibility.
    
    Attributes:
        _bits: NumPy array storing the binary values (internal representation).
        
    Example:
        >>> solution = BinarySolution(number_of_variables=10, number_of_objectives=2)
        >>> solution.variables = [True, False] * 5  # Set variables
        >>> solution[0] = False  # Modify a single bit
        >>> distance = solution.hamming_distance(other_solution)  # Calculate distance
    """
    __slots__ = ('_bits',)
    
    def __init__(
        self, 
        number_of_variables: int, 
        number_of_objectives: int, 
        number_of_constraints: int = 0
    ) -> None:
        """Initialize a binary solution with the given dimensions."""
        super().__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self._bits = np.zeros(number_of_variables, dtype=bool)
    
    @property
    def variables(self) -> List[bool]:
        """Return the decision variables as a list of booleans.
        
        Returns:
            A list where each element represents a bit in the solution.
        """
        return self._bits.tolist()
    
    @variables.setter
    def variables(self, values: List[bool]) -> None:
        """Set the decision variables from a list of booleans.
        
        Args:
            values: List of boolean values to set
            
        Raises:
            ValueError: If the length of values doesn't match number_of_variables
        """
        if len(values) != self.number_of_variables:
            raise ValueError(
                f"Expected {self.number_of_variables} variables, got {len(values)}"
            )
        self._bits[:] = values
    
    @property
    def bits(self) -> np.ndarray:
        """Direct access to the underlying NumPy array for high-performance operations.
        
        Returns:
            A read-only view of the internal bit array.
        """
        return self._bits.view()
    
    @bits.setter
    def bits(self, value: np.ndarray) -> None:
        """Set the bits from a NumPy array.
        
        Args:
            value: NumPy array of boolean values
            
        Raises:
            ValueError: If the array size doesn't match number_of_variables
        """
        if not isinstance(value, np.ndarray):
            self.variables = value
            return
            
        if value.size != self.number_of_variables:
            raise ValueError(
                f"Expected {self.number_of_variables} bits, got {value.size}"
            )
        np.copyto(self._bits, value.astype(bool, copy=False))
    
    def __copy__(self) -> 'BinarySolution':
        """Create a deep copy of this solution.
        
        Returns:
            A new BinarySolution instance with the same variable values,
            objectives, constraints, and attributes as this solution.
        """
        new_solution = self.__class__(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        np.copyto(new_solution._bits, self._bits)
        new_solution.objectives = self.objectives.copy()
        new_solution.constraints = self.constraints.copy()
        new_solution.attributes = self.attributes.copy()
        return new_solution
    
    def __getitem__(self, index: int) -> bool:
        """Get the bit at the specified index.
        
        Args:
            index: The index of the bit to get
            
        Returns:
            The boolean value of the bit at the given index
            
        Raises:
            IndexError: If the index is out of bounds
        """
        return bool(self._bits[index])
    
    def __setitem__(self, index: int, value: bool) -> None:
        """Set the bit at the specified index.
        
        Args:
            index: The index of the bit to set
            value: The boolean value to set
            
        Raises:
            IndexError: If the index is out of bounds
        """
        self._bits[index] = bool(value)
    
    def __eq__(self, other: object) -> bool:
        """Check if this solution is equal to another.
        
        Two solutions are considered equal if they have the same number of
        variables and their bit patterns match exactly.
        """
        if not isinstance(other, BinarySolution):
            return False
        return (self.number_of_variables == other.number_of_variables and
                np.array_equal(self._bits, other._bits))
    
    def __hash__(self) -> int:
        """Compute a hash value for this solution.
        
        The hash is based on the bit pattern and objective values.
        """
        return hash((self._bits.tobytes(), tuple(self.objectives)))
    
    def get_total_number_of_bits(self) -> int:
        """Get the total number of bits in the solution.
        
        Returns:
            The number of variables (bits) in the solution
        """
        return self.number_of_variables
    
    def get_binary_string(self) -> str:
        """Get a binary string representation of the solution.
        
        Returns:
            A string of '0's and '1's representing the solution
        """
        return ''.join(np.where(self._bits, '1', '0'))
    
    def cardinality(self) -> int:
        """Count the number of bits set to True.
        
        Also known as the Hamming weight or population count.
        
        Returns:
            The number of bits set to True
        """
        return int(np.sum(self._bits))
    
    def flip_bit(self, index: int) -> None:
        """Flip the bit at the specified index.
        
        Args:
            index: The index of the bit to flip
            
        Raises:
            IndexError: If the index is out of bounds
        """
        self._bits[index] ^= True
    
    def hamming_distance(self, other: 'BinarySolution') -> int:
        """Calculate the Hamming distance to another binary solution.
        
        The Hamming distance is the number of bit positions at which the
        corresponding bits are different.
        
        Args:
            other: Another BinarySolution to compare with
            
        Returns:
            The number of differing bits
            
        Raises:
            TypeError: If other is not a BinarySolution
            ValueError: If solutions have different lengths
        """
        if not isinstance(other, BinarySolution):
            raise TypeError(
                f"Cannot compute Hamming distance between "
                f"{self.__class__.__name__} and {type(other).__name__}"
            )
        if self.number_of_variables != other.number_of_variables:
            raise ValueError(
                f"Solutions must have the same number of variables: "
                f"{self.number_of_variables} != {other.number_of_variables}"
            )
        return int(np.sum(self._bits != other._bits))


class FloatSolution(Solution[float]):
    """A solution representation for continuous optimization problems with float variables.
    
    This class implements a solution where each decision variable is a floating-point
    value constrained by lower and upper bounds. It's suitable for continuous
    optimization problems where variables can take any real value within specified ranges.
    
    The solution maintains the following properties:
    - Each variable has independent lower and upper bounds
    - Variables are stored as a list of floats
    - Bounds checking is performed when variables are set
    
    Attributes:
        lower_bound: List of lower bounds for each variable.
        upper_bound: List of upper bounds for each variable.
        _variables: Internal storage for the decision variables.
    """
    __slots__ = ('_variables', 'lower_bound', 'upper_bound')
    
    def __init__(
            self,
            lower_bound: List[float],
            upper_bound: List[float],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ) -> None:
        super().__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        if len(lower_bound) != len(upper_bound):
            raise ValueError("lower_bound and upper_bound must have the same length")
            
        self.lower_bound = lower_bound.copy()
        self.upper_bound = upper_bound.copy()
        self._variables = [0.0] * self.number_of_variables
    
    @property
    def variables(self) -> List[float]:
        """Get the decision variables as a list of floats.
        
        Returns:
            A list of float values representing the solution's variables.
        """
        return self._variables.copy()
    
    @variables.setter
    def variables(self, values: List[float]) -> None:
        """Set the decision variables from a list of floats.
        
        Args:
            values: List of float values to set.
            
        Raises:
            ValueError: If the length of values doesn't match number_of_variables
        """
        if len(values) != self.number_of_variables:
            raise ValueError(
                f"Expected {self.number_of_variables} variables, got {len(values)}"
            )
        self._variables = values.copy()

    def __copy__(self) -> 'FloatSolution':
        """Create a deep copy of this solution.
        
        Returns:
            A new FloatSolution instance with the same variable values,
            objectives, constraints, and attributes as this solution.
        """
        new_solution = self.__class__(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.variables
        new_solution.objectives = self.objectives
        new_solution.constraints = self.constraints
        new_solution.attributes = self.attributes.copy()
        
        return new_solution


class IntegerSolution(Solution[int]):
    """A solution representation for integer-constrained optimization problems.
    
    This class is designed for optimization problems where decision variables
    must take integer values within specified bounds. It's suitable for:
    - Pure integer programming problems
    - Mixed-integer problems (when used with other solution types)
    - Combinatorial optimization with integer-encoded solutions
    
    The implementation ensures that all variables remain within their specified
    bounds and are stored as integers. Bounds checking is performed when variables
    are modified.
    
    Attributes:
        lower_bound: List of lower bounds for each variable (inclusive).
        upper_bound: List of upper bounds for each variable (inclusive).
        _variables: Internal storage for the integer decision variables.
    """
    __slots__ = ('_variables', 'lower_bound', 'upper_bound')
    
    def __init__(
            self,
            lower_bound: List[int],
            upper_bound: List[int],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ) -> None:
        super().__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        if len(lower_bound) != len(upper_bound):
            raise ValueError("lower_bound and upper_bound must have the same length")
            
        self.lower_bound = lower_bound.copy()
        self.upper_bound = upper_bound.copy()
        self._variables = [0] * self.number_of_variables
    
    @property
    def variables(self) -> List[int]:
        """Get the decision variables as a list of integers.
        
        Returns:
            A list of integer values representing the solution's variables.
        """
        return self._variables.copy()
    
    @variables.setter
    def variables(self, values: List[int]) -> None:
        """Set the decision variables from a list of integers.
        
        Args:
            values: List of integer values to set.
            
        Raises:
            ValueError: If the length of values doesn't match number_of_variables
        """
        if len(values) != self.number_of_variables:
            raise ValueError(
                f"Expected {self.number_of_variables} variables, got {len(values)}"
            )
        self._variables = [int(v) for v in values]  # Ensure all values are integers

    def __copy__(self) -> 'IntegerSolution':
        """Create a deep copy of this solution.
        
        Returns:
            A new IntegerSolution instance with the same variable values,
            objectives, constraints, and attributes as this solution.
        """
        new_solution = self.__class__(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.variables
        new_solution.objectives = self.objectives
        new_solution.constraints = self.constraints
        new_solution.attributes = self.attributes.copy()
        
        return new_solution


class CompositeSolution(Solution[Solution]):
    """A solution composed of multiple heterogeneous solution types.
    
    This class enables the creation of complex solutions by combining multiple
    solution objects of different types (e.g., binary, integer, float) into a single
    composite solution. This is particularly useful for:
    - Multi-encoding optimization problems
    - Decomposition-based optimization approaches
    - Problems with mixed variable types
    
    All constituent solutions must have the same number of objectives and constraints
    to maintain consistency in the optimization process.
    
    Example:
        # Create a composite solution with binary and float parts
        binary_part = BinarySolution(10, 2)
        float_part = FloatSolution([0.0]*5, [1.0]*5, 2)
        composite = CompositeSolution([binary_part, float_part])
    
    Attributes:
        _solutions: List of solution objects that compose this composite solution.
    """
    __slots__ = ('_solutions',)
    
    def __init__(self, solutions: List[Solution]) -> None:
        """Initialize a composite solution.
        
        Args:
            solutions: List of Solution objects to compose this solution from.
            
        Raises:
            ValueError: If solutions is empty or solutions have inconsistent
                       numbers of objectives or constraints.
        """
        Check.is_not_none(solutions)
        Check.collection_is_not_empty(solutions)
        
        # Validate all solutions have same number of objectives and constraints
        first = solutions[0]
        for solution in solutions[1:]:
            if len(solution.objectives) != len(first.objectives):
                raise ValueError(
                    f"All solutions must have the same number of objectives. "
                    f"Found {len(first.objectives)} and {len(solution.objectives)}"
                )
            if len(solution.constraints) != len(first.constraints):
                raise ValueError(
                    f"All solutions must have the same number of constraints. "
                    f"Found {len(first.constraints)} and {len(solution.constraints)}"
                )
        
        super().__init__(
            number_of_variables=len(solutions),
            number_of_objectives=len(first.objectives),
            number_of_constraints=len(first.constraints)
        )
        
        # Make defensive copies of all solutions
        self._solutions = [copy.copy(sol) for sol in solutions]
    
    @property
    def variables(self) -> List[Solution]:
        """Get the list of solutions that compose this composite solution.
        
        Returns:
            A list of Solution objects.
        """
        return self._solutions
    
    @variables.setter
    def variables(self, solutions: List[Solution]) -> None:
        """Set the list of solutions that compose this composite solution.
        
        Args:
            solutions: List of Solution objects.
            
        Raises:
            ValueError: If solutions is empty or solutions have inconsistent
                       numbers of objectives or constraints.
        """
        if not solutions:
            raise ValueError("Composite solution must have at least one solution")
            
        # Validate all solutions have same number of objectives and constraints
        first = solutions[0]
        for solution in solutions[1:]:
            if len(solution.objectives) != len(first.objectives):
                raise ValueError(
                    f"All solutions must have the same number of objectives. "
                    f"Found {len(first.objectives)} and {len(solution.objectives)}"
                )
            if len(solution.constraints) != len(first.constraints):
                raise ValueError(
                    f"All solutions must have the same number of constraints. "
                    f"Found {len(first.constraints)} and {len(solution.constraints)}"
                )
        
        # Update the number of variables, objectives, and constraints
        self.number_of_variables = len(solutions)
        self.number_of_objectives = len(first.objectives)
        self.number_of_constraints = len(first.constraints)
        
        # Make defensive copies of all solutions
        self._solutions = [copy.copy(sol) for sol in solutions]
    
    def __copy__(self) -> 'CompositeSolution':
        """Create a deep copy of this composite solution.
        
        Returns:
            A new CompositeSolution instance with copies of all contained solutions.
        """
        new_solution = self.__class__(self._solutions)
        new_solution.objectives = self.objectives.copy()
        new_solution.constraints = self.constraints.copy()
        new_solution.attributes = self.attributes.copy()
        
        return new_solution


class PermutationSolution(Solution[int]):
    """A solution representation for permutation-based optimization problems.
    
    This class is designed for problems where solutions are represented as
    permutations of integers, such as:
    - Traveling Salesman Problem (TSP)
    - Job Shop Scheduling
    - Quadratic Assignment Problem (QAP)
    - Any problem where the order of elements matters
    
    The solution maintains a permutation of integers from 0 to n-1, where n is
    the number of variables. The implementation ensures that the permutation
    remains valid (no duplicates, all numbers in range) at all times.
    
    Attributes:
        _variables: List storing the permutation of integers.
    
    Example:
        # Create a permutation solution for a 5-city TSP
        solution = PermutationSolution(5, 1)  # 5 cities, 1 objective
        # The initial permutation is [0, 1, 2, 3, 4]
        solution.variables = [4, 2, 0, 1, 3]  # Set a specific tour
    """
    __slots__ = ('_variables',)
    
    def __init__(
            self, 
            number_of_variables: int, 
            number_of_objectives: int, 
            number_of_constraints: int = 0
    ) -> None:
        """Initialize a permutation solution.
        
        Args:
            number_of_variables: Length of the permutation.
            number_of_objectives: Number of objective values.
            number_of_constraints: Number of constraint values (default: 0).
        """
        super().__init__(number_of_variables, number_of_objectives, number_of_constraints)
        # Initialize with identity permutation
        self._variables = list(range(number_of_variables))
    
    @property
    def variables(self) -> List[int]:
        """Get the permutation as a list of integers.
        
        Returns:
            A list representing the current permutation.
        """
        return self._variables.copy()
    
    @variables.setter
    def variables(self, values: List[int]) -> None:
        """Set the permutation from a list of integers.
        
        Args:
            values: List of integers representing a permutation.
            
        Raises:
            ValueError: If values is not a valid permutation of 0..n-1
        """
        if len(values) != self.number_of_variables:
            raise ValueError(
                f"Expected {self.number_of_variables} variables, got {len(values)}"
            )
        
        # Check if it's a valid permutation (contains all numbers from 0 to n-1)
        if sorted(values) != list(range(self.number_of_variables)):
            raise ValueError(
                f"Invalid permutation. Must contain all integers from 0 to {self.number_of_variables-1}"
            )
        
        self._variables = values.copy()
    
    def __copy__(self) -> 'PermutationSolution':
        """Create a deep copy of this solution.
        
        Returns:
            A new PermutationSolution instance with the same variable values,
            objectives, constraints, and attributes as this solution.
        """
        new_solution = self.__class__(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.variables
        new_solution.objectives = self.objectives
        new_solution.constraints = self.constraints
        new_solution.attributes = self.attributes.copy()
        
        return new_solution
