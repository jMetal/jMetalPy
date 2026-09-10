from unittest.mock import Mock, patch

import numpy as np
import pytest

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    Solution,
)
from jmetal.operator.mutation import (
    BitFlipMutation,
    CompositeMutation,
    IntegerPolynomialMutation,
    PolynomialMutation,
)
from jmetal.util.ckecking import (
    EmptyCollectionException,
)


class TestPolynomialMutation:
    @pytest.mark.parametrize("probability", [-1, 1.1])
    def test_should_constructor_raise_exception_for_invalid_probability(self, probability):
        with pytest.raises(ValueError):
            PolynomialMutation(probability=probability)

    def test_should_constructor_create_a_valid_operator(self):
        operator = PolynomialMutation(probability=0.5)
        assert operator.probability == 0.5
        assert operator.distribution_index == 20.0

    @patch("random.random", return_value=0.1)  # Force mutation
    def test_should_mutate_when_probability_is_one(self, _):
        solution = FloatSolution([0.0], [1.0], 1)
        solution.variables = [0.5]

        operator = PolynomialMutation(probability=1.0)
        mutated_solution = operator.execute(solution)

        assert mutated_solution.variables[0] != 0.5

    @patch("random.random", return_value=0.9)  # Skip mutation
    def test_should_not_mutate_when_probability_is_zero(self, _):
        solution = FloatSolution([0.0], [1.0], 1)
        solution.variables = [0.5]

        operator = PolynomialMutation(probability=0.0)
        mutated_solution = operator.execute(solution)

        assert mutated_solution.variables[0] == 0.5


class TestBitFlipMutation:
    def test_should_mutate_binary_solution(self):
        # Create a binary solution with known bits
        solution = BinarySolution(number_of_variables=4, number_of_objectives=1)
        solution.variables = [True, False, True, False]
        original_bits = solution.variables.copy()

        # Inject an rng to control which bits flip (all of them here)
        mock_rng = Mock()
        mock_rng.random.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        operator = BitFlipMutation(probability=1.0, rng=mock_rng)

        mutated = operator.execute(solution)

        # Verify mutation
        assert isinstance(mutated, BinarySolution)
        assert len(mutated.variables) == len(original_bits)

        # All bits should have flipped
        assert all(not orig == mut for orig, mut in zip(original_bits, mutated.variables))

    def test_should_not_mutate_with_zero_probability(self):
        # Create a binary solution with known bits
        solution = BinarySolution(number_of_variables=4, number_of_objectives=1)
        solution.variables = [True, False, True, False]
        original_bits = solution.variables.copy()

        # Inject an rng that would flip every bit if probability allowed it
        mock_rng = Mock()
        mock_rng.random.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        operator = BitFlipMutation(probability=0.0, rng=mock_rng)

        mutated = operator.execute(solution)

        # Verify no mutation occurred
        assert mutated.variables == original_bits

    def test_rng_produces_deterministic_mutation_for_a_fixed_seed(self):
        solution = BinarySolution(number_of_variables=20, number_of_objectives=1)
        solution.variables = [True] * 20

        operator_a = BitFlipMutation(probability=0.5, rng=np.random.default_rng(42))
        operator_b = BitFlipMutation(probability=0.5, rng=np.random.default_rng(42))

        mutated_a = operator_a.execute(solution.__copy__())
        mutated_b = operator_b.execute(solution.__copy__())

        assert mutated_a.variables == mutated_b.variables

    def test_without_rng_falls_back_to_a_fresh_default_generator(self):
        # Guards that omitting rng still produces a usable, self-contained operator
        # instead of erroring or silently reusing global numpy state across instances.
        solution = BinarySolution(number_of_variables=4, number_of_objectives=1)
        solution.variables = [True, False, True, False]

        operator = BitFlipMutation(probability=1.0)
        mutated = operator.execute(solution)

        assert isinstance(mutated, BinarySolution)
        assert isinstance(operator.rng, np.random.Generator)


class TestIntegerPolynomialMutation:
    def test_should_keep_values_within_bounds(self):
        # Test with a solution at lower bound
        lower_solution = IntegerSolution([0], [10], 1)
        lower_solution.variables = [0]

        # Test with a solution at upper bound
        upper_solution = IntegerSolution([0], [10], 1)
        upper_solution.variables = [10]

        # Test with a solution in the middle
        mid_solution = IntegerSolution([0], [10], 1)
        mid_solution.variables = [5]

        operator = IntegerPolynomialMutation(probability=1.0, distribution_index=20.0)

        # Run multiple times to ensure stability
        for _ in range(10):
            # Test lower bound
            mutated = operator.execute(lower_solution)
            assert mutated.variables[0] >= 0
            assert isinstance(mutated.variables[0], int)

            # Test upper bound
            mutated = operator.execute(upper_solution)
            assert mutated.variables[0] <= 10
            assert isinstance(mutated.variables[0], int)

            # Test middle value
            mutated = operator.execute(mid_solution)
            assert 0 <= mutated.variables[0] <= 10
            assert isinstance(mutated.variables[0], int)


class TestCompositeMutation:
    @pytest.fixture
    def mock_mutation(self):
        class MockMutation(Mutation[Solution]):
            def __init__(self):
                super().__init__()
                self.called = False

            def execute(self, solution: Solution) -> Solution:
                self.called = True
                return solution

            def get_name(self) -> str:
                return "MockMutation"

        return MockMutation()

    def test_should_apply_all_mutations_in_sequence(self):
        # Create a composite solution with two float solutions
        from jmetal.core.solution import FloatSolution
        from jmetal.operator.mutation import Mutation

        # Create two float solutions
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 1)
        solution1.variables = [0.5, 0.5]

        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 1)
        solution2.variables = [0.3, 0.7]

        composite_solution = CompositeSolution([solution1, solution2])

        # Create mock mutations that are subclasses of Mutation
        class MockMutation(Mutation[FloatSolution]):
            def __init__(self, return_solution):
                super().__init__(probability=1.0)
                self.return_solution = return_solution
                self.called = False

            def execute(self, solution):
                self.called = True
                return self.return_solution

            def get_name(self):
                return "MockMutation"

        # Create mock mutations that will return the original solutions
        mock_mutation1 = MockMutation(solution1)
        mock_mutation2 = MockMutation(solution2)

        # Create and execute composite mutation
        composite = CompositeMutation([mock_mutation1, mock_mutation2])
        result = composite.execute(composite_solution)

        # Verify mocks were called
        assert mock_mutation1.called
        assert mock_mutation2.called

        # Verify the result is a composite solution with the correct structure
        assert isinstance(result, CompositeSolution)
        assert len(result.variables) == 2
        # Compare variable values instead of solution objects
        assert result.variables[0].variables == solution1.variables
        assert result.variables[1].variables == solution2.variables

    def test_should_raise_exception_for_empty_mutation_list(self):
        with pytest.raises(EmptyCollectionException):
            CompositeMutation([])
