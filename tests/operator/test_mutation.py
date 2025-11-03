import random
from typing import List
from unittest.mock import patch

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
    solution = BinarySolution(number_of_variables=6, number_of_objectives=1)
    solution.bits = np.array([True, False, False, True, True, False])
    return solution

@pytest.fixture
def float_solution():
    solution = FloatSolution(
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 1.0],
        number_of_objectives=2
    )
    solution.variables = [0.5, 0.5]
    return solution

@pytest.fixture
def integer_solution():
    solution = IntegerSolution(
        lower_bound=[0, 0],
        upper_bound=[10, 10],
        number_of_objectives=2
    )
    solution.variables = [5, 5]
    return solution

class TestPolynomialMutation:
    @pytest.mark.parametrize("probability", [-1, 1.1])
    def test_should_constructor_raise_exception_for_invalid_probability(self, probability):
        with pytest.raises(ValueError):
            PolynomialMutation(probability=probability)

    def test_should_constructor_create_a_valid_operator(self):
        operator = PolynomialMutation(probability=0.5)
        assert operator.probability == 0.5
        assert operator.distribution_index == 20.0

    @patch('random.random', return_value=0.1)  # Force mutation
    def test_should_mutate_when_probability_is_one(self, _):
        solution = FloatSolution([0.0], [1.0], 1)
        solution.variables = [0.5]
        
        operator = PolynomialMutation(probability=1.0)
        mutated_solution = operator.execute(solution)
        
        assert mutated_solution.variables[0] != 0.5

    @patch('random.random', return_value=0.9)  # Skip mutation
    def test_should_not_mutate_when_probability_is_zero(self, _):
        solution = FloatSolution([0.0], [1.0], 1)
        solution.variables = [0.5]
        
        operator = PolynomialMutation(probability=0.0)
        mutated_solution = operator.execute(solution)
        
        assert mutated_solution.variables[0] == 0.5

class TestBitFlipMutation:
    def test_should_mutate_binary_solution(self, binary_solution):
        operator = BitFlipMutation(probability=1.0)
        # Store original bits
        original_bits = binary_solution.bits.copy()
        mutated = operator.execute(binary_solution)
        
        assert isinstance(mutated, BinarySolution)
        assert len(mutated.bits) == len(original_bits)
        # Check that at least one bit has flipped
        assert any(orig != mut for orig, mut in zip(original_bits, mutated.bits))

    def test_should_not_mutate_with_zero_probability(self, binary_solution):
        operator = BitFlipMutation(probability=0.0)
        original_bits = binary_solution.bits.copy()
        mutated = operator.execute(binary_solution)
        
        assert np.array_equal(mutated.bits, original_bits)

class TestIntegerPolynomialMutation:
    @pytest.mark.parametrize("value,expected", [
        (0, 0),    # Lower bound
        (10, 10),  # Upper bound
        (5, 5)     # Middle
    ])
    def test_should_keep_integer_values_within_bounds(self, value, expected):
        solution = IntegerSolution([0], [10], 1)
        solution.variables = [value]
        
        operator = IntegerPolynomialMutation(probability=1.0)
        mutated = operator.execute(solution)
        
        assert mutated.variables[0] == expected

class TestCompositeMutation:
    def test_should_apply_all_mutations_in_sequence(self, float_solution):
        # Create a composite solution with two float solutions
        composite_solution = CompositeSolution([float_solution, float_solution])
        
        # Create a mock that returns a new solution with mutated values
        def mock_execute(solution):
            # Create a new solution with mutated values
            result = FloatSolution(
                lower_bound=solution.lower_bound,
                upper_bound=solution.upper_bound,
                number_of_objectives=solution.number_of_objectives
            )
            result.variables = [v + 0.1 for v in solution.variables]
            return result

        with patch.object(PolynomialMutation, 'execute', side_effect=mock_execute) as mock_mutation:
            operator = CompositeMutation([PolynomialMutation(1.0), PolynomialMutation(1.0)])
            result = operator.execute(composite_solution)
            
            # Should be called once for each solution in the composite
            assert mock_mutation.call_count == 2
            # Verify the result was mutated
            assert result.variables[0].variables[0] == pytest.approx(float_solution.variables[0] + 0.1)
            assert result.variables[1].variables[1] == pytest.approx(float_solution.variables[1] + 0.1)

    def test_should_raise_exception_for_empty_mutation_list(self):
        with pytest.raises(EmptyCollectionException):
            CompositeMutation([])