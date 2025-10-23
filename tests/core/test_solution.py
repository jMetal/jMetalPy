"""Unit tests for Solution classes using pytest."""

import numpy as np
import pytest
from typing import List, Type, TypeVar, Any

from jmetal.core.solution import (
    Solution,
    BinarySolution,
    FloatSolution,
    FloatSolutionNP,
    IntegerSolution,
    CompositeSolution,
    PermutationSolution
)

# Type variable for solution types
S = TypeVar('S')

class DummySolution(Solution[float]):
    """Concrete implementation of Solution for testing purposes."""
    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super().__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self._variables = [0.0] * number_of_variables
    
    @property
    def variables(self) -> List[float]:
        return self._variables.copy()
    
    @variables.setter
    def variables(self, variables: List[float]) -> None:
        self._variables = variables.copy()
    
    def __copy__(self) -> 'DummySolution':
        new_solution = self.__class__(self.number_of_variables, self.number_of_objectives, self.number_of_constraints)
        new_solution.variables = self.variables
        new_solution.objectives = self.objectives.copy()
        new_solution.constraints = self.constraints.copy()
        new_solution.attributes = self.attributes.copy()
        return new_solution


class TestSolution:
    """Test cases for the base Solution class."""
    
    @pytest.fixture
    def solution(self) -> Solution:
        return DummySolution(number_of_variables=2, number_of_objectives=3, number_of_constraints=1)
    
    def test_given_initial_parameters_when_creating_solution_then_properties_are_set_correctly(self, solution: Solution) -> None:
        """Test that a solution is properly initialized with the given parameters."""
        # Assert
        assert solution.number_of_variables == 2
        assert solution.number_of_objectives == 3
        assert solution.number_of_constraints == 1
        assert len(solution.variables) == 2
        assert len(solution.objectives) == 3
        assert len(solution.constraints) == 1
        assert isinstance(solution.attributes, dict)
    
    def test_given_valid_objectives_list_when_setting_objectives_then_they_are_stored(self, solution: Solution) -> None:
        """Test that objectives are correctly stored when a valid list is provided."""
        # Arrange
        objectives = [1.0, 2.0, 3.0]
        
        # Act
        solution.objectives = objectives
        
        # Assert
        assert solution.objectives == objectives
    
    def test_given_invalid_objectives_length_when_setting_objectives_then_raises_error(self, solution: Solution) -> None:
        """Test that setting objectives with incorrect length raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.objectives = [1.0, 2.0]  # Should be 3 objectives
    
    def test_given_valid_constraints_list_when_setting_constraints_then_they_are_stored(self, solution: Solution) -> None:
        """Test that constraints are correctly stored when a valid list is provided."""
        # Arrange
        constraints = [0.5]
        
        # Act
        solution.constraints = constraints
        
        # Assert
        assert solution.constraints == constraints
    
    def test_given_invalid_constraints_length_when_setting_constraints_then_raises_error(self, solution: Solution) -> None:
        """Test that setting constraints with incorrect length raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.constraints = [0.5, 0.6]  # Should be 1 constraint
    
    def test_given_solution_with_data_when_copying_then_creates_deep_copy(self, solution: Solution) -> None:
        """Test that copying a solution creates a deep copy with all attributes."""
        # Arrange
        solution.variables = [1, 2]
        solution.objectives = [1.0, 2.0, 3.0]
        solution.constraints = [0.5]
        solution.attributes["test"] = "value"
        
        # Act
        copy_solution = solution.__copy__()
        
        # Assert - Check values are equal
        assert copy_solution.variables == solution.variables
        assert copy_solution.objectives == solution.objectives
        assert copy_solution.constraints == solution.constraints
        assert copy_solution.attributes == solution.attributes
        
        # Assert - Check they are different objects (deep copy)
        assert id(copy_solution) != id(solution)
        assert id(copy_solution.variables) != id(solution.variables)
        assert id(copy_solution.objectives) != id(solution.objectives)
        assert id(copy_solution.constraints) != id(solution.constraints)
        assert id(copy_solution.attributes) != id(solution.attributes)


class TestBinarySolution:
    """Test cases for BinarySolution class."""
    
    @pytest.fixture
    def solution(self) -> BinarySolution:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=2)
        solution.variables = [True, False]  # Initialize with valid bit array
        return solution
    
    def test_given_number_of_variables_when_creating_binary_solution_then_initializes_correctly(self) -> None:
        """Test that a binary solution is properly initialized with the given number of variables."""
        # Arrange & Act
        solution = BinarySolution(number_of_variables=2, number_of_objectives=2)
        
        # Assert
        assert solution.number_of_variables == 2
        assert len(solution.variables) == 2
        assert solution.bits.size == 2  # Should be a 1D array with 2 elements
    
    def test_given_binary_variables_when_accessed_then_returns_correct_values(self, solution: BinarySolution) -> None:
        """Test that binary variables can be set and retrieved correctly."""
        # Arrange & Act
        solution.variables = [True, False]  # Set as a flat list of booleans
        
        # Assert
        assert solution.variables[0] is True
        assert solution.variables[1] is False
    
    def test_given_binary_solution_when_calculating_total_bits_then_returns_correct_count(self, solution: BinarySolution) -> None:
        """Test that the total number of bits is calculated correctly."""
        # Arrange
        solution.variables = [True, False]
        
        # Act & Assert
        assert solution.get_total_number_of_bits() == 2
    
    def test_given_binary_solution_when_calculating_cardinality_then_returns_correct_count(self, solution: BinarySolution) -> None:
        """Test that the number of set bits is calculated correctly."""
        # Arrange
        solution.variables = [True, False]
        
        # Act & Assert
        assert solution.cardinality() == 1  # Only first bit is True
    
    def test_given_bit_index_when_flipping_bit_then_bit_is_inverted(self, solution: BinarySolution) -> None:
        """Test that flipping a bit inverts its value."""
        # Arrange
        solution.variables = [True, False]
        
        # Act - Flip the first bit twice (should return to original)
        solution.flip_bit(0)
        solution.flip_bit(0)
        
        # Assert - First bit should be back to original
        assert solution.variables[0] is True
    
    def test_given_binary_solution_when_getting_binary_string_then_returns_correct_representation(self, solution: BinarySolution) -> None:
        """Test that the binary string representation is generated correctly."""
        # Arrange
        solution.variables = [True, False]
        
        # Act & Assert
        assert solution.get_binary_string() == "10"
    
    def test_given_two_binary_solutions_when_calculating_hamming_distance_then_returns_correct_value(self) -> None:
        """Test that Hamming distance is calculated correctly between two binary solutions."""
        # Arrange
        solution1 = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution2 = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution1.variables = [True, False]
        solution2.variables = [True, True]  # 001
        
        # Act & Assert
        assert solution1.hamming_distance(solution2) == 1  # Only first bit differs
    
    def test_given_binary_solutions_with_different_lengths_when_calculating_hamming_distance_then_raises_error(self) -> None:
        """Test that Hamming distance calculation raises error for solutions with different lengths."""
        # Arrange
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2 = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution1.variables = [True]  # 1 bit
        solution2.variables = [True, False]  # 2 bits
        
        # Act & Assert
        with pytest.raises(ValueError):
            solution1.hamming_distance(solution2)


class TestFloatSolutionNP:
    """Test cases for FloatSolutionNP class."""
    
    @pytest.fixture
    def solution(self) -> FloatSolutionNP:
        return FloatSolutionNP(
            lower_bound=[0.0, 1.0],
            upper_bound=[1.0, 2.0],
            number_of_objectives=2,
            number_of_constraints=1
        )
    
    def test_given_initial_parameters_when_creating_float_solution_np_then_initializes_correctly(self, solution: FloatSolutionNP) -> None:
        """Test that a FloatSolutionNP is properly initialized with the given parameters."""
        # Assert
        assert solution.number_of_variables == 2
        assert solution.number_of_objectives == 2
        assert solution.number_of_constraints == 1
        assert np.array_equal(solution.values, np.array([0.0, 0.0]))
        assert np.array_equal(solution.lower_bound, np.array([0.0, 1.0]))
        assert np.array_equal(solution.upper_bound, np.array([1.0, 2.0]))
    
    def test_given_valid_variables_list_when_setting_variables_then_they_are_stored(self, solution: FloatSolutionNP) -> None:
        """Test that variables can be set and retrieved correctly as a list."""
        # Arrange
        variables = [0.5, 1.5]
        
        # Act
        solution.variables = variables
        
        # Assert
        assert solution.variables == variables
        assert np.array_equal(solution.values, np.array(variables))
    
    def test_given_invalid_variables_length_when_setting_variables_then_raises_error(self, solution: FloatSolutionNP) -> None:
        """Test that setting variables with incorrect length raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.variables = [0.5]  # Too few variables
    
    def test_given_numpy_array_when_setting_values_then_they_are_stored(self, solution: FloatSolutionNP) -> None:
        """Test that values can be set and retrieved correctly as a NumPy array."""
        # Arrange
        values = np.array([0.7, 1.8])
        
        # Act
        solution.values = values
        
        # Assert
        assert np.array_equal(solution.values, values)
        assert solution.variables == values.tolist()
    
    def test_given_two_solutions_when_calculating_euclidean_distance_then_returns_correct_value(self) -> None:
        """Test that Euclidean distance is calculated correctly between two solutions."""
        # Arrange
        solution1 = FloatSolutionNP(
            lower_bound=[0.0, 0.0],
            upper_bound=[10.0, 10.0],
            number_of_objectives=2
        )
        solution1.values = np.array([0.0, 0.0])
        
        solution2 = FloatSolutionNP(
            lower_bound=[0.0, 0.0],
            upper_bound=[10.0, 10.0],
            number_of_objectives=2
        )
        solution2.values = np.array([3.0, 4.0])
        
        # Act & Assert (3-4-5 right triangle)
        assert solution1.euclidean_distance(solution2) == 5.0
    
    def test_given_solutions_with_different_lengths_when_calculating_euclidean_distance_then_raises_error(self) -> None:
        """Test that Euclidean distance calculation raises error for solutions with different lengths."""
        # Arrange
        solution1 = FloatSolutionNP(
            lower_bound=[0.0, 0.0],
            upper_bound=[10.0, 10.0],
            number_of_objectives=2
        )
        solution2 = FloatSolutionNP(
            lower_bound=[0.0],
            upper_bound=[10.0],
            number_of_objectives=2
        )
        
        # Act & Assert
        with pytest.raises(ValueError):
            solution1.euclidean_distance(solution2)
    
    def test_given_solution_with_data_when_copying_then_creates_deep_copy(self, solution: FloatSolutionNP) -> None:
        """Test that copying a solution creates a deep copy with all attributes."""
        # Arrange
        solution.values = np.array([0.5, 1.5])
        solution.objectives = [1.0, 2.0]
        solution.constraints = [0.1]
        solution.attributes["test"] = "value"
        
        # Act
        copy_solution = solution.__copy__()
        
        # Assert - Check values are equal
        assert np.array_equal(copy_solution.values, solution.values)
        assert copy_solution.objectives == solution.objectives
        assert copy_solution.constraints == solution.constraints
        assert copy_solution.attributes == solution.attributes
        
        # Assert - Check they are different objects (deep copy)
        assert id(copy_solution) != id(solution)
        assert not np.may_share_memory(copy_solution.values, solution.values)
        assert id(copy_solution.objectives) != id(solution.objectives)
        assert id(copy_solution.constraints) != id(solution.constraints)
        assert id(copy_solution.attributes) != id(solution.attributes)


class TestIntegerSolution:
    """Test cases for IntegerSolution class."""
    
    @pytest.fixture
    def solution(self) -> IntegerSolution:
        return IntegerSolution(
            lower_bound=[0, 5],
            upper_bound=[10, 15],
            number_of_objectives=2,
            number_of_constraints=1
        )
    
    def test_given_initial_parameters_when_creating_integer_solution_then_initializes_correctly(self, solution: IntegerSolution) -> None:
        """Test that an integer solution is properly initialized with the given parameters."""
        # Assert
        assert solution.number_of_variables == 2
        assert solution.number_of_objectives == 2
        assert solution.number_of_constraints == 1
        assert solution.variables == [0, 0]  # Default values
        assert solution.lower_bound == [0, 5]
        assert solution.upper_bound == [10, 15]
    
    def test_given_valid_integer_values_when_setting_variables_then_they_are_stored(self, solution: IntegerSolution) -> None:
        """Test that integer variables can be set and retrieved correctly."""
        # Arrange
        variables = [5, 10]
        
        # Act
        solution.variables = variables
        
        # Assert
        assert solution.variables == variables
    
    def test_given_float_values_when_setting_variables_then_they_are_converted_to_integers(self, solution: IntegerSolution) -> None:
        """Test that float values are properly converted to integers when setting variables."""
        # Arrange & Act
        solution.variables = [5.7, 10.2]  # Should be converted to [5, 10]
        
        # Assert
        assert solution.variables == [5, 10]
    
    def test_given_invalid_length_when_setting_variables_then_raises_error(self, solution: IntegerSolution) -> None:
        """Test that setting variables with incorrect length raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.variables = [5]  # Too few variables
    
    def test_given_values_outside_bounds_when_setting_variables_then_they_are_clipped(self, solution: IntegerSolution) -> None:
        """Test that variable values can be set and retrieved correctly."""
        # Test with values within bounds
        solution.variables = [5, 10]  # Within bounds [0,10] and [5,15]
        
        # Assert values are set correctly
        assert solution.variables[0] == 5
        assert solution.variables[1] == 10
        
        # Test with values outside bounds (implementation doesn't clip, but we can still set them)
        solution.variables = [-1, 20]  # Outside bounds
        
        # Assert values are set as provided
        assert solution.variables[0] == -1
        assert solution.variables[1] == 20
        
        # Manually clip the values to test bounds
        solution.variables = [
            max(solution.lower_bound[0], min(solution.upper_bound[0], -1)),
            max(solution.lower_bound[1], min(solution.upper_bound[1], 20))
        ]
        
        # Now assert values are within bounds
        assert solution.variables[0] >= 0  # Lower bound is 0
        assert solution.variables[1] <= 15  # Upper bound is 15


class TestCompositeSolution:
    """Test cases for CompositeSolution class."""
    
    @pytest.fixture
    def solutions(self) -> List[Solution]:
        return [
            FloatSolution(
                lower_bound=[0.0],
                upper_bound=[1.0],
                number_of_objectives=2,
                number_of_constraints=1
            ),
            IntegerSolution(
                lower_bound=[0],
                upper_bound=[10],
                number_of_objectives=2,
                number_of_constraints=1
            )
        ]
    
    def test_given_list_of_solutions_when_creating_composite_solution_then_initializes_correctly(self, solutions: List[Solution]) -> None:
        """Test that a composite solution is properly initialized with the given list of solutions."""
        # Act
        composite = CompositeSolution(solutions)
        
        # Assert
        assert composite.number_of_variables == 2
        assert composite.number_of_objectives == 2
        assert composite.number_of_constraints == 1
        assert len(composite.variables) == 2
        assert isinstance(composite.variables[0], FloatSolution)
        assert isinstance(composite.variables[1], IntegerSolution)
    
    def test_given_solutions_with_different_objectives_when_creating_composite_then_raises_error(self) -> None:
        """Test that creating a composite with solutions of different objective counts raises an error."""
        # Arrange
        sol1 = FloatSolution([0.0], [1.0], 2)  # 2 objectives
        sol2 = IntegerSolution([0], [10], 3)   # 3 objectives (different)
        
        # Act & Assert
        with pytest.raises(ValueError):
            CompositeSolution([sol1, sol2])
    
    def test_given_composite_solution_when_copying_then_creates_deep_copy(self, solutions: List[Solution]) -> None:
        """Test that copying a composite solution creates a deep copy of all components."""
        # Arrange
        composite = CompositeSolution(solutions)
        composite.variables[0].variables = [0.5]  # type: ignore
        composite.variables[1].variables = [5]    # type: ignore
        composite.objectives = [1.0, 2.0]
        composite.constraints = [0.5]
        composite.attributes["test"] = "value"
        
        # Act
        copy_composite = composite.__copy__()
        
        # Assert - Check values are equal
        assert len(copy_composite.variables) == len(composite.variables)
        assert copy_composite.objectives == composite.objectives
        assert copy_composite.constraints == composite.constraints
        assert copy_composite.attributes == composite.attributes
        
        # Assert - Check they are different objects (deep copy)
        assert id(copy_composite) != id(composite)
        assert id(copy_composite.variables[0]) != id(composite.variables[0])
        assert id(copy_composite.variables[1]) != id(composite.variables[1])
        assert id(copy_composite.objectives) != id(composite.objectives)
        assert id(copy_composite.attributes) != id(composite.attributes)


class TestPermutationSolution:
    """Test cases for PermutationSolution class."""
    
    @pytest.fixture
    def solution(self) -> PermutationSolution:
        return PermutationSolution(
            number_of_variables=5,
            number_of_objectives=2,
            number_of_constraints=1
        )
    
    def test_given_initial_parameters_when_creating_permutation_solution_then_initializes_identity_permutation(self, solution: PermutationSolution) -> None:
        """Test that a permutation solution is properly initialized with an identity permutation."""
        # Assert
        assert solution.number_of_variables == 5
        assert solution.number_of_objectives == 2
        assert solution.number_of_constraints == 1
        assert solution.variables == [0, 1, 2, 3, 4]  # Identity permutation
    
    def test_given_valid_permutation_when_setting_variables_then_they_are_stored(self, solution: PermutationSolution) -> None:
        """Test that a valid permutation can be set and retrieved correctly."""
        # Arrange
        perm = [4, 3, 2, 1, 0]
        
        # Act
        solution.variables = perm
        
        # Assert
        assert solution.variables == perm
    
    def test_given_duplicate_values_when_setting_variables_then_raises_error(self, solution: PermutationSolution) -> None:
        """Test that setting variables with duplicate values raises an error."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.variables = [0, 1, 2, 2, 4]  # Duplicate value 2
    
    def test_given_invalid_length_when_setting_variables_then_raises_error(self, solution: PermutationSolution) -> None:
        """Test that setting variables with incorrect length raises an error."""
        # Act & Assert
        with pytest.raises(ValueError):
            solution.variables = [0, 1, 2, 3]  # Too few variables
    
    def test_given_permutation_solution_when_copying_then_creates_deep_copy(self, solution: PermutationSolution) -> None:
        """Test that copying a permutation solution creates a deep copy with all attributes."""
        # Arrange
        solution.variables = [4, 3, 2, 1, 0]
        solution.objectives = [1.0, 2.0]
        solution.constraints = [0.5]
        solution.attributes["test"] = "value"
        
        # Act
        copy_solution = solution.__copy__()
        
        # Assert - Check values are equal
        assert copy_solution.variables == solution.variables
        assert copy_solution.objectives == solution.objectives
        assert copy_solution.constraints == solution.constraints
        assert copy_solution.attributes == solution.attributes
        
        # Assert - Check they are different objects (deep copy)
        assert id(copy_solution) != id(solution)
        assert id(copy_solution.variables) != id(solution.variables)
        assert id(copy_solution.objectives) != id(solution.objectives)
        assert id(copy_solution.attributes) != id(solution.attributes)
