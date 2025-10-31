"""Tests for mutation operators.

This module contains tests for all mutation operators in jMetalPy.
"""
import random
import unittest
from typing import List

import numpy as np
import pytest

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
)
from jmetal.operator.mutation import (
    BitFlipMutation,
    CompositeMutation,
    IntegerPolynomialMutation,
    PolynomialMutation,
    SimpleRandomMutation,
    UniformMutation,
    LevyFlightMutation,
    PowerLawMutation,
)
from jmetal.util.ckecking import (
    EmptyCollectionException,
    InvalidConditionException,
    NoneParameterException,
)

# Fixtures

@pytest.fixture
def binary_solution():
    """Create a binary solution for testing."""
    solution = BinarySolution(number_of_variables=4, number_of_objectives=1)
    solution.bits = np.array([True, False, True, False], dtype=bool)
    return solution


class TestPolynomialMutation:
    """Tests for PolynomialMutation operator."""
    
    @pytest.mark.parametrize("invalid_probability", [-1, 1.1, -0.1, 2.0])
    def test_raises_exception_for_invalid_probability(self, invalid_probability):
        """Test that constructor raises ValueError for invalid probability values."""
        with pytest.raises(ValueError):
            PolynomialMutation(probability=invalid_probability)
    
    def test_creates_valid_operator(self):
        """Test that constructor creates a valid operator with correct parameters."""
        # When
        operator = PolynomialMutation(probability=0.5, distribution_index=20)
        
        # Then
        assert operator.probability == 0.5
        assert operator.distribution_index == 20
        assert isinstance(operator, Mutation)
    
    @pytest.mark.parametrize("invalid_distribution_index", [-1, -0.1, -10])
    def test_raises_exception_for_negative_distribution_index(self, invalid_distribution_index):
        """Test that constructor raises ValueError for negative distribution index."""
        with pytest.raises(ValueError):
            PolynomialMutation(0.5, -1)

    @pytest.fixture
    def float_solution(self):
        """Create a float solution for testing."""
        solution = FloatSolution(
            lower_bound=[0.0, 0.0, 0.0],
            upper_bound=[1.0, 1.0, 1.0],
            number_of_objectives=1,
            number_of_constraints=0
        )
        solution.variables = [0.5, 0.5, 0.5]
        return solution

    def test_solution_unchanged_with_zero_probability(self, float_solution):
        """Test that solution remains unchanged when probability is zero."""
        # Given
        operator = PolynomialMutation(probability=0.0)
        original_vars = float_solution.variables.copy()
        
        # When
        mutated_solution = operator.execute(float_solution)
        
        # Then
        assert mutated_solution.variables == original_vars

    def test_solution_changes_with_probability_one(self, float_solution):
        """Test that solution changes when mutation probability is 1.0."""
        # Given
        operator = PolynomialMutation(probability=1.0, distribution_index=20)
        original_vars = float_solution.variables.copy()
        
        # When/Then - Try multiple times due to randomness
        changed = False
        for _ in range(10):
            float_solution.variables = original_vars.copy()
            mutated_solution = operator.execute(float_solution)
            
            if mutated_solution.variables != original_vars:
                changed = True
                # Verify all variables are within bounds
                assert all(0 <= x <= 1 for x in mutated_solution.variables), \
                    f"Variable out of bounds: {mutated_solution.variables}"
                break
                
        assert changed, "Solution should change when mutation probability is 1.0"

    def test_works_with_float_solution_subclass(self):
        """Test that mutation works with FloatSolution subclasses."""
        # Given: A custom FloatSolution subclass
        class CustomFloatSolution(FloatSolution):
            """Custom FloatSolution subclass for testing."""
            
            def __init__(
                self,
                lower_bound: List[float],
                upper_bound: List[float],
                number_of_objectives: int,
                number_of_constraints: int = 0,
            ):
                super().__init__(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    number_of_objectives=number_of_objectives,
                    number_of_constraints=number_of_constraints
                )
                self.custom_attr = "test"
        
        # Given: Test data
        operator = PolynomialMutation(probability=1.0, distribution_index=20.0)
        solution = CustomFloatSolution(
            lower_bound=[-5, -5, -5],
            upper_bound=[5, 5, 5],
            number_of_objectives=2,
            number_of_constraints=0
        )
        original_vars = [1.0, 2.0, 3.0]
        solution.variables = original_vars.copy()
        
        # When/Then: Try multiple times due to randomness
        changed = False
        for _ in range(10):
            solution.variables = original_vars.copy()
            mutated_solution = operator.execute(solution)
            
            # Check if any variable changed
            if any(x != y for x, y in zip(original_vars, mutated_solution.variables)):
                changed = True
                # Verify all variables are within bounds
                assert all(-5 <= x <= 5 for x in mutated_solution.variables), \
                    f"Variable out of bounds: {mutated_solution.variables}"
                # Verify custom attributes are preserved
                assert hasattr(mutated_solution, 'custom_attr')
                assert mutated_solution.custom_attr == "test"
                break
                
        assert changed, "Solution should change when mutation probability is 1.0"


class TestBitFlipMutation:
    """Tests for BitFlipMutation operator."""
    
    @pytest.fixture
    def binary_solution(self):
        """Create a binary solution for testing."""
        solution = BinarySolution(number_of_variables=4, number_of_objectives=1)
        solution.bits = np.array([True, False, True, False], dtype=bool)
        return solution
    
    @pytest.mark.parametrize("invalid_probability", [-1, 1.1, -0.1, 2.0])
    def test_raises_exception_for_invalid_probability(self, invalid_probability):
        """Test that constructor raises ValueError for invalid probability values."""
        with pytest.raises(ValueError):
            BitFlipMutation(probability=invalid_probability)
    
    def test_creates_valid_operator(self):
        """Test that constructor creates a valid operator with correct probability."""
        # When
        operator = BitFlipMutation(probability=0.5)
        
        # Then
        assert operator.probability == 0.5
        assert isinstance(operator, Mutation)
    
    def test_solution_unchanged_with_zero_probability(self, binary_solution):
        """Test that solution remains unchanged when probability is zero."""
        # Given
        operator = BitFlipMutation(probability=0.0)
        original_bits = binary_solution.bits.copy()
        
        # When
        mutated_solution = operator.execute(binary_solution)
        
        # Then
        assert np.array_equal(mutated_solution.bits, original_bits)
    
    def test_solution_changes_with_probability_one(self, binary_solution):
        """Test that solution changes when mutation probability is 1.0."""
        # Given
        operator = BitFlipMutation(probability=1.0)
        original_bits = binary_solution.bits.copy()
        
        # Set random seed for deterministic test
        np.random.seed(42)
        try:
            # When
            mutated_solution = operator.execute(binary_solution)
            
            # Then
            assert not np.array_equal(mutated_solution.bits, original_bits)
            assert len(mutated_solution.bits) == 4
        finally:
            # Reset random seed
            np.random.seed(None)
    
    def test_solution_has_same_length_after_mutation(self, binary_solution):
        """Test that mutation preserves the solution length."""
        # Given
        operator = BitFlipMutation(probability=1.0)
        original_length = len(binary_solution.bits)
        
        # When
        mutated_solution = operator.execute(binary_solution)
        
        # Then
        assert len(mutated_solution.bits) == original_length


class TestUniformMutation:
    """Tests for UniformMutation operator."""
    
    @pytest.fixture
    def float_solution(self):
        """Create a float solution for testing."""
        solution = FloatSolution(
            lower_bound=[0, 0],
            upper_bound=[1, 1],
            number_of_objectives=1,
            number_of_constraints=0
        )
        solution.variables = [0.5, 0.5]
        return solution
    
    @pytest.mark.parametrize("probability,perturbation,expected_error", [
        (-1, 0.5, "probability must be in \\[0, 1\\]"),
        (1.1, 0.5, "probability must be in \\[0, 1\\]"),
        (0.1, -0.5, "perturbation must be positive"),
        (0.1, 0.0, "perturbation must be positive")
    ])
    def test_raises_exception_for_invalid_parameters(self, probability, perturbation, expected_error):
        """Test that constructor raises ValueError for invalid parameters."""
        with pytest.raises(ValueError, match=expected_error):
            UniformMutation(probability=probability, perturbation=perturbation)
    
    def test_creates_valid_operator(self):
        """Test that constructor creates a valid operator with correct parameters."""
        # When
        operator = UniformMutation(probability=0.1, perturbation=0.5)
        
        # Then
        assert operator.probability == 0.1
        assert operator.perturbation == 0.5
        assert isinstance(operator, Mutation)
    
    def test_solution_unchanged_with_zero_probability(self, float_solution):
        """Test that solution remains unchanged when probability is zero."""
        # Given
        operator = UniformMutation(probability=0.0, perturbation=0.5)
        original_vars = float_solution.variables.copy()
        
        # When
        mutated_solution = operator.execute(float_solution)
        
        # Then
        assert mutated_solution.variables == original_vars
    
    def test_solution_changes_with_probability_one(self, float_solution):
        """Test that solution changes when mutation probability is 1.0."""
        # Given
        operator = UniformMutation(probability=1.0, perturbation=0.5)
        original_vars = float_solution.variables.copy()
        
        # Set random seed for deterministic test
        np.random.seed(42)
        try:
            # When
            mutated_solution = operator.execute(float_solution)
            
            # Then
            assert mutated_solution.variables != original_vars
            assert len(mutated_solution.variables) == 2
            assert all(0 <= x <= 1 for x in mutated_solution.variables)
        finally:
            # Reset random seed
            np.random.seed(None)
    
    def test_variables_stay_within_bounds(self, float_solution):
        """Test that mutated variables stay within their bounds."""
        # Given: A solution with bounds [0, 1] for both variables
        operator = UniformMutation(probability=1.0, perturbation=1.0)
        
        # When: Mutate with high perturbation
        np.random.seed(42)
        try:
            mutated_solution = operator.execute(float_solution)
            
            # Then: All variables should be within [0, 1]
            assert all(0 <= x <= 1 for x in mutated_solution.variables), \
                f"Variables out of bounds: {mutated_solution.variables}"
        finally:
            np.random.seed(None)


class TestIntegerPolynomialMutation:
    """Tests for IntegerPolynomialMutation operator."""
    
    @pytest.fixture
    def integer_solution(self):
        """Create an integer solution for testing."""
        solution = IntegerSolution(
            lower_bound=[0, 0, 0],
            upper_bound=[5, 5, 5],
            number_of_objectives=1,
            number_of_constraints=0
        )
        solution.variables = [1, 2, 3]
        return solution
    
    @pytest.mark.parametrize("probability,expected_error", [
        (-1, "The probability is lower than zero: -1"),
        (1.1, "The probability is greater than one: 1.1")
    ])
    def test_raises_exception_for_invalid_probability(self, probability, expected_error):
        """Test that constructor raises Exception for invalid probability values."""
        with pytest.raises(Exception, match=expected_error):
            IntegerPolynomialMutation(probability=probability)
    
    def test_creates_valid_operator(self):
        """Test that constructor creates a valid operator with correct probability."""
        # When
        operator = IntegerPolynomialMutation(probability=0.5)
        
        # Then
        assert operator.probability == 0.5
        assert isinstance(operator, Mutation)
    
    def test_solution_unchanged_with_zero_probability(self, integer_solution):
        """Test that solution remains unchanged when probability is zero."""
        # Given
        operator = IntegerPolynomialMutation(probability=0.0)
        original_vars = integer_solution.variables.copy()
        
        # When
        mutated_solution = operator.execute(integer_solution)
        
        # Then
        assert mutated_solution.variables == original_vars
    
    def test_solution_changes_with_probability_one(self, integer_solution):
        """Test that solution changes when mutation probability is 1.0."""
        # Given
        operator = IntegerPolynomialMutation(probability=1.0, distribution_index=20.0)
        original_vars = integer_solution.variables.copy()
        
        # Set random seed for deterministic test
        np.random.seed(42)
        try:
            # When/Then - Try multiple times due to randomness
            changed = False
            for _ in range(50):  # Increased from 10 to 50 attempts
                integer_solution.variables = original_vars.copy()
                mutated_solution = operator.execute(integer_solution)
                
                # Debug output
                print(f"Original: {original_vars}, Mutated: {mutated_solution.variables}")
                
                if mutated_solution.variables != original_vars:
                    changed = True
                    # Verify all variables are within bounds and integers
                    assert all(isinstance(x, int) for x in mutated_solution.variables), \
                        f"Non-integer value found: {mutated_solution.variables}"
                    assert all(0 <= x <= 5 for x in mutated_solution.variables), \
                        f"Variable out of bounds: {mutated_solution.variables}"
                    break
            
            if not changed:
                # If we get here, all attempts failed - run one more time with debug info
                integer_solution.variables = original_vars.copy()
                mutated_solution = operator.execute(integer_solution)
                print(f"Final attempt - Original: {original_vars}, Mutated: {mutated_solution.variables}")
                
            assert changed, ("Solution should change when mutation probability is 1.0. "
                           f"Original: {original_vars}, Mutated: {mutated_solution.variables}")
        finally:
            # Reset random seed
            np.random.seed(None)

    def test_raises_exception_if_parameter_list_is_empty(self):
        """Test that constructor raises EmptyCollectionException if parameter list is empty."""
        with pytest.raises(EmptyCollectionException):
            CompositeMutation([])

    def test_creates_valid_operator_with_single_mutation_operator(self):
        """Test that constructor creates a valid operator with a single mutation operator."""
        # Given
        mutation: Mutation = PolynomialMutation(0.9, 20.0)

        # When
        operator = CompositeMutation([mutation])

        # Then
        assert operator is not None
        assert len(operator.mutation_operators_list) == 1
        assert operator.mutation_operators_list[0] == mutation

    def test_creates_valid_operator_with_multiple_mutation_operators(self):
        """Test that constructor creates a valid operator with multiple mutation operators."""
        # Given
        polynomial_mutation = PolynomialMutation(1.0, 20.0)
        bit_flip_mutation = BitFlipMutation(0.01)

        # When
        operator = CompositeMutation([polynomial_mutation, bit_flip_mutation])

        # Then
        assert operator is not None
        assert len(operator.mutation_operators_list) == 2
        assert isinstance(operator.mutation_operators_list[0], PolynomialMutation)
        assert isinstance(operator.mutation_operators_list[1], BitFlipMutation)

    def test_execute_raises_exception_for_invalid_solution_type(self):
        """Test that execute raises an exception for invalid solution type."""
        # Given
        operator = CompositeMutation([BitFlipMutation(0.5)])
        solution = FloatSolution([0.0], [1.0], 1, 1)

        # When/Then
        with pytest.raises((InvalidConditionException, TypeError)):
            operator.execute(solution)


class TestLevyFlightMutation:
    """Tests for LevyFlightMutation operator."""
    
    @pytest.fixture
    def solution(self):
        lower = [0.0, -5.0, 10.0]
        upper = [1.0, 5.0, 20.0]
        sol = FloatSolution(lower, upper, 1)
        sol.variables = [0.5, 0.0, 15.0]
        return sol
    
    def test_mutation_probability_zero(self, solution):
        """Test that solution remains unchanged when mutation probability is 0."""
        # Given
        def repair(val, low, up):
            return max(min(val, up), low)
            
        mutation = LevyFlightMutation(mutation_probability=0.0, beta=1.5, step_size=0.5, repair_operator=repair)
        original_vars = solution.variables.copy()
        
        # When
        mutated_solution = mutation.execute(solution)
        
        # Then
        assert mutated_solution.variables == original_vars

    def test_mutation_probability_one(self, solution):
        """Test that solution changes when mutation probability is 1."""
        # Given
        def repair(val, low, up):
            return max(min(val, up), low)
            
        mutation = LevyFlightMutation(mutation_probability=1.0, beta=1.5, step_size=0.5, repair_operator=repair)
        original_vars = solution.variables.copy()
        
        # When
        mutated_solution = mutation.execute(solution)
        
        # Then
        assert mutated_solution.variables != original_vars, \
            "At least one variable should change when mutation probability is 1.0"

    def test_beta_parameter(self, solution):
        """Test that beta parameter is set correctly and solution stays within bounds."""
        # Given/When
        def repair(val, low, up):
            return max(min(val, up), low)
            
        mutation = LevyFlightMutation(mutation_probability=1.0, beta=1.9, step_size=0.5, repair_operator=repair)
        mutated = mutation.execute(solution)
        
        # Then
        assert mutation.beta == 1.9
        for i, (lower, upper) in enumerate(zip([0.0, -5.0, 10.0], [1.0, 5.0, 20.0])):
            assert lower <= mutated.variables[i] <= upper, \
                f"Variable at index {i} out of bounds: {mutated.variables[i]}"

    def test_step_size_parameter(self, solution):
        """Test that step_size parameter is set correctly and solution stays within bounds."""
        # Given/When
        def repair(val, low, up):
            return max(min(val, up), low)
            
        mutation = LevyFlightMutation(mutation_probability=1.0, beta=1.5, step_size=0.2, repair_operator=repair)
        mutated = mutation.execute(solution)
        
        # Then
        assert mutation.step_size == 0.2
        for i, (lower, upper) in enumerate(zip([0.0, -5.0, 10.0], [1.0, 5.0, 20.0])):
            assert lower <= mutated.variables[i] <= upper, \
                f"Variable at index {i} out of bounds: {mutated.variables[i]}"

    def test_repair_operator(self, solution):
        """Test that the repair operator is applied correctly."""
        # Given
        def repair(val, low, up):
            return max(min(val, up), low)
            
        # When
        mutation = LevyFlightMutation(
            mutation_probability=1.0, 
            beta=1.5, 
            step_size=10.0,  # Large step to ensure out of bounds
            repair_operator=repair
        )
        mutated = mutation.execute(solution)
        
        # Then
        for i, (lower, upper) in enumerate(zip([0.0, -5.0, 10.0], [1.0, 5.0, 20.0])):
            assert lower <= mutated.variables[i] <= upper, \
                f"Variable at index {i} out of bounds: {mutated.variables[i]}"

    def test_to_string(self, solution):
        """Test the string representation of the mutation operator."""
        # Given
        def repair(val, low, up):
            return max(min(val, up), low)
            
        mutation = LevyFlightMutation(mutation_probability=1.0, beta=1.5, step_size=0.5, repair_operator=repair)
        
        # When/Then - The actual string representation includes the full module path
        assert "LevyFlightMutation" in str(mutation)


class TestPowerLawMutation(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        lower = [0.0, 0.0]
        upper = [10.0, 5.0]
        from jmetal.core.solution import FloatSolution
        self.solution = FloatSolution(lower, upper, number_of_objectives=1)
        self.solution.variables = [5.0, 2.5]

    def test_default_constructor(self):
        mutation = PowerLawMutation()
        self.assertEqual(mutation.probability, 0.01)
        self.assertEqual(mutation.delta, 1.0)

    def test_custom_parameters(self):
        mutation = PowerLawMutation(0.05, 2.0)
        self.assertEqual(mutation.probability, 0.05)
        self.assertEqual(mutation.delta, 2.0)

    def test_invalid_probability(self):
        with self.assertRaises(Exception):
            PowerLawMutation(-0.1, 1.0)
        with self.assertRaises(Exception):
            PowerLawMutation(1.1, 1.0)

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            PowerLawMutation(0.01, 0.0)
        with self.assertRaises(ValueError):
            PowerLawMutation(0.01, -1.0)

    def test_execute_returns_solution(self):
        mutation = PowerLawMutation(0.0, 1.0)
        result = mutation.execute(self.solution)
        self.assertIs(result, self.solution)

    def test_execute_zero_probability(self):
        mutation = PowerLawMutation(0.0, 1.0)
        original = self.solution.variables.copy()
        mutation.execute(self.solution)
        self.assertEqual(self.solution.variables, original)

    def test_execute_always_mutate(self):
        mutation = PowerLawMutation(1.0, 1.0)
        original_variables = self.solution.variables.copy()
        
        # Try multiple times to account for randomness
        changed = False
        for _ in range(10):  # Try up to 10 times
            self.solution.variables = original_variables.copy()
            mutation.execute(self.solution)
            if any(x != y for x, y in zip(original_variables, self.solution.variables)):
                changed = True
                break
                
        self.assertTrue(changed, "Solution should change when mutation probability is 1.0")

    def test_extreme_random_values(self):
        mutation = PowerLawMutation(1.0, 1.0)
        orig_random = random.random
        random.random = lambda: 1e-15
        mutation.execute(self.solution)
        random.random = lambda: 1.0 - 1e-15
        mutation.execute(self.solution)
        random.random = orig_random

    def test_repair_operator(self):
        def repair(val, low, up):
            return low if val < low else up if val > up else val

        mutation = PowerLawMutation(1.0, 1.0, repair_operator=repair)
        mutation.execute(self.solution)

    def test_multiple_variables(self):
        mutation = PowerLawMutation(1.0, 1.0)
        from jmetal.core.solution import FloatSolution
        lower = [0.0, 0.0, 0.0]
        upper = [10.0, 10.0, 10.0]
        sol = FloatSolution(lower, upper, number_of_objectives=1)
        sol.variables = [5.0, 3.0, 7.0]
        mutation.execute(sol)
        self.assertEqual(len(sol.variables), 3)

    def test_str(self):
        mutation = PowerLawMutation(0.05, 2.0)
        self.assertIn("Power Law mutation", mutation.get_name())
        

if __name__ == "__main__":
    unittest.main()
