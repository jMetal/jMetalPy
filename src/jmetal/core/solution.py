from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, final, Dict, Any
import numpy as np

from jmetal.util.ckecking import Check

BitSet = List[bool]
S = TypeVar("S")


class Solution(Generic[S], ABC):
    """
    Abstract base class for optimization solutions.
    
    Subclasses must implement the variables property to provide
    their own storage mechanism (lists, NumPy arrays, etc.).
    """
    
    def __init__(
        self, 
        number_of_variables: int, 
        number_of_objectives: int, 
        number_of_constraints: int = 0
    ) -> None:
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        
        # Initialize with None - subclasses should set appropriate types
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
    """
    High-performance binary solution using NumPy.
    
    This class represents a binary solution where each variable is a single bit,
    stored efficiently in a NumPy array. It provides both a Pythonic interface
    through the `variables` property and high-performance NumPy operations
    through the `bits` property.
    
    Args:
        number_of_variables: Number of binary variables (bits) in the solution
        number_of_objectives: Number of objective values
        number_of_constraints: Number of constraint values (default: 0)
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
    """Class representing float solutions
    
    This class implements a solution where each variable is a float value
    constrained by lower and upper bounds.
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


@final
class FloatSolutionNP(Solution[float]):
    """High-performance float solution using NumPy."""
    __slots__ = ('_values', '_lower_bound', '_upper_bound', 
                 'objectives', 'constraints', 'attributes',
                 'number_of_variables', 'number_of_objectives', 'number_of_constraints')
    
    def __init__(
        self,
        lower_bound: np.ndarray | List[float],
        upper_bound: np.ndarray | List[float],
        number_of_objectives: int,
        number_of_constraints: int = 0
    ) -> None:
        number_of_variables = len(lower_bound)
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        
        self._lower_bound = np.asarray(lower_bound, dtype=np.float64)
        self._upper_bound = np.asarray(upper_bound, dtype=np.float64)
        self._values = np.zeros(number_of_variables, dtype=np.float64)
        
        self.objectives = [0.0] * number_of_objectives
        self.constraints = [0.0] * number_of_constraints
        self.attributes = {}
    
    @property
    def variables(self) -> List[float]:
        """Return values as a list of floats."""
        return self._values.tolist()
    
    @variables.setter
    def variables(self, values: List[float]) -> None:
        """Set values from a list of floats."""
        if len(values) != self.number_of_variables:
            raise ValueError(f"Expected {self.number_of_variables} variables, got {len(values)}")
        self._values[:] = values
    
    @property
    def values(self) -> np.ndarray:
        """Direct access to the NumPy array."""
        return self._values
    
    @values.setter
    def values(self, arr: np.ndarray) -> None:
        """Set values from a NumPy array."""
        if arr.size != self.number_of_variables:
            raise ValueError(f"Expected {self.number_of_variables} values")
        np.copyto(self._values, arr.astype(np.float64, copy=False))
    
    @property
    def lower_bound(self) -> np.ndarray:
        """Get the lower bounds as a NumPy array."""
        return self._lower_bound
    
    @property
    def upper_bound(self) -> np.ndarray:
        """Get the upper bounds as a NumPy array."""
        return self._upper_bound
    
    def __copy__(self) -> 'FloatSolutionNP':
        new_solution = FloatSolutionNP(
            self._lower_bound,
            self._upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        np.copyto(new_solution._values, self._values)
        new_solution.objectives = self.objectives[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution
    
    def __getitem__(self, index: int) -> float:
        """Get a variable by index."""
        return float(self._values[index])
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set a variable by index."""
        self._values[index] = value
    
    def __eq__(self, other: object) -> bool:
        """Check if this solution is equal to another."""
        if not isinstance(other, FloatSolutionNP):
            return False
        return (self.number_of_variables == other.number_of_variables and
                np.allclose(self._values, other._values))
    
    def euclidean_distance(self, other: 'FloatSolutionNP') -> float:
        """Calculate Euclidean distance to another solution.
        
        Args:
            other: Another FloatSolutionNP to compare with
            
        Returns:
            The Euclidean distance between the solutions
            
        Raises:
            TypeError: If other is not a FloatSolutionNP
            ValueError: If solutions have different number of variables
        """
        if not isinstance(other, FloatSolutionNP):
            raise TypeError(f"Expected FloatSolutionNP, got {type(other).__name__}")
        if self.number_of_variables != other.number_of_variables:
            raise ValueError("Solutions must have the same number of variables")
        return float(np.linalg.norm(self._values - other._values))


class IntegerSolution(Solution[int]):
    """Class representing integer solutions
    
    This class implements a solution where each variable is an integer value
    constrained by lower and upper bounds.
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
    """Class representing solutions composed of a list of solutions.
    
    This class allows creating mixed solutions by combining solutions of different types.
    All solutions in the composite must have the same number of objectives and constraints.
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
    """Class representing permutation solutions.
    
    This class implements a solution where variables represent a permutation
    of integers from 0 to number_of_variables-1.
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
