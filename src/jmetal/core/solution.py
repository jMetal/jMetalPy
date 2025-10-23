from abc import ABC
from typing import Generic, List, TypeVar, final
import numpy as np

from jmetal.util.ckecking import Check

BitSet = List[bool]
S = TypeVar("S")


class Solution(Generic[S], ABC):
    """Class representing solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.variables = [[] for _ in range(number_of_variables)]
        self.objectives = [0.0 for _ in range(number_of_objectives)]
        self.constraints = [0.0 for _ in range(number_of_constraints)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return "Solution(variables={},objectives={},constraints={})".format(
            self.variables, self.objectives, self.constraints
        )


class BinarySolution(Solution[BitSet]):
    """Class representing float solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

        self.bits_per_variable = []

    def __copy__(self):
        new_solution = BinarySolution(len(self.variables), len(self.objectives), len(self.constraints))
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()
        new_solution.bits_per_variable = self.bits_per_variable

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total

    def get_binary_string(self) -> str:
        string = ""
        for bit in self.variables[0]:
            string += "1" if bit else "0"
        return string

    def cardinality(self, variable_index) -> int:
        return sum(1 for _ in self.variables[variable_index] if _)


@final
class BinarySolutionNP(Solution[bool]):
    """
    High-performance binary solution implementation using NumPy.
    Each variable is a single bit, stored efficiently in a NumPy array.
    
    Attributes:
        number_of_variables: Number of bits/variables in the solution
        number_of_objectives: Number of objective functions
        number_of_constraints: Number of constraints (default: 0)
    """
    __slots__ = ('_bits', 'objectives', 'constraints', 'attributes',
                'number_of_variables', 'number_of_objectives', 'number_of_constraints')
    
    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        """Initialize the solution."""
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self._bits = np.zeros(number_of_variables, dtype=np.bool_)
        self.objectives = [0.0] * number_of_objectives
        self.constraints = [0.0] * number_of_constraints
        self.attributes = {}
        
    @property
    def variables(self):
        """Return the variables as a list for compatibility with existing code."""
        return [self._bits.tolist()]
    
    def __copy__(self) -> 'BinarySolutionNP':
        """Create a deep copy of this solution."""
        new_solution = self.__class__(
            number_of_variables=self.number_of_variables,
            number_of_objectives=len(self.objectives),
            number_of_constraints=len(self.constraints)
        )
        
        # Efficiently copy the bits array
        np.copyto(new_solution._bits, self._bits)
        
        # Copy other attributes
        new_solution.objectives = self.objectives.copy()
        new_solution.constraints = self.constraints.copy()
        new_solution.attributes = self.attributes.copy()
        
        return new_solution
        
    @property
    def bits(self) -> np.ndarray:
        """Return a view of the bits."""
        return self._bits.view()
        
    @bits.setter
    def bits(self, value):
        """Set the bits from a NumPy array or boolean list."""
        if isinstance(value, np.ndarray):
            if value.size != self.number_of_variables:
                raise ValueError(f"Expected {self.number_of_variables} bits, got {value.size}")
            np.copyto(self._bits, value.astype(np.bool_))
        else:
            if len(value) != self.number_of_variables:
                raise ValueError(f"Expected {self.number_of_variables} bits, got {len(value)}")
            self._bits[:] = [bool(v) for v in value]
        
    def get_total_number_of_bits(self) -> int:
        """Return the total number of bits (same as number of variables)."""
        return self.number_of_variables
        
    def get_binary_string(self) -> str:
        """Return the binary string representation using efficient string operations."""
        return ''.join(np.where(self._bits, '1', '0'))
        
    def cardinality(self, variable_index: int = None) -> int:
        """
        Return the number of bits set to True.
        If variable_index is provided, returns the value of that bit (0 or 1).
        """
        if variable_index is not None:
            return int(self._bits[variable_index])
        return int(np.sum(self._bits))
        
    def flip_bit(self, index: int) -> None:
        """Flip the bit at the specified index using in-place XOR."""
        self._bits[index] ^= True
        
    def get_bits_as_int(self) -> int:
        """Convert the binary solution to an integer using bit shifting."""
        return int(''.join(np.where(self._bits, '1', '0')), 2)
        
    def set_bits_from_int(self, value: int) -> None:
        """Set the bits from an integer value using vectorized operations."""
        if value < 0:
            raise ValueError("Value must be non-negative")
        max_val = (1 << self.number_of_variables) - 1
        if value > max_val:
            value = value & max_val  # Truncate to available bits
            
        # Fast bit extraction using bitwise operations
        self._bits[:] = [(value >> i) & 1 for i in reversed(range(self.number_of_variables))]
        
    def hamming_distance(self, other: 'BinarySolutionNP') -> int:
        """Calculate the Hamming distance to another binary solution."""
        if not isinstance(other, BinarySolutionNP):
            raise TypeError(f"Expected BinarySolutionNP, got {type(other).__name__}")
        if self.number_of_variables != other.number_of_variables:
            raise ValueError("Solutions must have the same number of variables")
        return int(np.sum(self._bits != other._bits))
        
    def __getitem__(self, index: int) -> bool:
        """Efficient bit access."""
        return bool(self._bits[index])
        
    def __setitem__(self, index: int, value: bool) -> None:
        """Efficient bit assignment."""
        self._bits[index] = bool(value)
        
    def __str__(self) -> str:
        """Efficient string representation using f-strings."""
        return (f"BinarySolutionNP(bits={self.get_binary_string()}, "
                f"objectives={self.objectives}, "
                f"constraints={self.constraints})")
                
    def __eq__(self, other: object) -> bool:
        """Efficient equality comparison."""
        if not isinstance(other, BinarySolutionNP):
            return False
        return (np.array_equal(self._bits, other._bits) and
                self.objectives == other.objectives and
                self.constraints == other.constraints)
                
    def __hash__(self) -> int:
        """Compute hash based on bits and objectives."""
        return hash((self._bits.tobytes(), tuple(self.objectives)))


class FloatSolution(Solution[float]):
    """Class representing float solutions"""

    def __init__(
            self,
            lower_bound: List[float],
            upper_bound: List[float],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, len(self.objectives), len(self.constraints)
        )
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.lower_bound = self.lower_bound
        new_solution.upper_bound = self.upper_bound

        new_solution.attributes = self.attributes.copy()

        return new_solution


class IntegerSolution(Solution[int]):
    """Class representing integer solutions"""

    def __init__(
            self,
            lower_bound: List[int],
            upper_bound: List[int],
            number_of_objectives: int,
            number_of_constraints: int = 0
    ):
        super(IntegerSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, len(self.objectives), len(self.constraints)
        )
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.lower_bound = self.lower_bound
        new_solution.upper_bound = self.upper_bound

        new_solution.attributes = self.attributes.copy()

        return new_solution


class CompositeSolution(Solution):
    """Class representing solutions composed of a list of solutions. The idea is that each decision  variable can
    be a solution of any type, so we can create mixed solutions (e.g., solutions combining any of the existing
    encodings). The adopted approach has the advantage of easing the reuse of existing variation operators, but all the
    solutions in the list will need to have the same function and constraint violation values.

    It is assumed that problems using instances of this class will properly manage the solutions it contains.
    """

    def __init__(self, solutions: List[Solution]):
        super(CompositeSolution, self).__init__(
            len(solutions), len(solutions[0].objectives), len(solutions[0].constraints)
        )
        Check.is_not_none(solutions)
        Check.collection_is_not_empty(solutions)

        for solution in solutions:
            Check.that(
                len(solution.objectives) == len(solutions[0].objectives),
                "The solutions in the list must have the same number of objectives: "
                + str(len(solutions[0].objectives)),
            )
            Check.that(
                len(solution.constraints) == len(solutions[0].constraints),
                "The solutions in the list must have the same number of constraints: "
                + str(len(solutions[0].constraints)),
            )

        self.variables = solutions

    def __copy__(self):
        new_solution = CompositeSolution(self.variables)

        new_solution.objectives = self.objectives[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """Class representing permutation solutions"""

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = PermutationSolution(len(self.variables), len(self.objectives), len(self.constraints))
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
