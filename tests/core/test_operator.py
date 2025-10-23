import numpy as np
import pytest
from unittest.mock import patch, MagicMock, ANY

from jmetal.core.operator import Mutation, Crossover
from jmetal.core.solution import BinarySolution
from jmetal.operator.mutation import BitFlipMutation
from jmetal.operator.crossover import SPXCrossover


class TestSPXCrossover:
    def test_given_valid_probability_when_initializing_then_probability_is_set_correctly(self):
        """Test initialization with valid probability values."""
        # Test with minimum probability
        crossover = SPXCrossover(0.0)
        assert crossover.probability == 0.0
        
        # Test with maximum probability
        crossover = SPXCrossover(1.0)
        assert crossover.probability == 1.0
        
        # Test with mid-range probability
        crossover = SPXCrossover(0.5)
        assert crossover.probability == 0.5

    def test_given_invalid_probability_when_initializing_then_raises_value_error(self):
        """Test initialization with invalid probability values."""
        # Test negative probability
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            SPXCrossover(-0.1)
            
        # Test probability > 1.0
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            SPXCrossover(1.1)

    def test_given_spx_crossover_when_getting_name_then_returns_correct_string(self):
        """Test the get_name method returns the correct string."""
        crossover = SPXCrossover(0.5)
        assert crossover.get_name() == "Single point crossover"

    def test_given_spx_crossover_when_getting_number_of_parents_then_returns_two(self):
        """Test get_number_of_parents returns 2."""
        crossover = SPXCrossover(0.5)
        assert crossover.get_number_of_parents() == 2

    def test_given_spx_crossover_when_getting_number_of_children_then_returns_two(self):
        """Test get_number_of_children returns 2."""
        crossover = SPXCrossover(0.5)
        assert crossover.get_number_of_children() == 2

    def create_mock_solution(self, bits, variables):
        """Helper method to create a mock BinarySolution with the given bits and variables."""
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = len(variables)
        solution.bits = bits.copy()
        solution.variables = [v.copy() for v in variables]
        return solution

    def test_given_probability_one_when_executing_then_always_performs_crossover(self):
        """Test that crossover is performed when random value is less than probability."""
        # Create mock solutions with bits
        parent1 = self.create_mock_solution(
            bits=np.array([0, 1, 0, 1], dtype=bool),
            variables=[np.array([0, 1], dtype=bool), np.array([0, 1], dtype=bool)]
        )
        parent2 = self.create_mock_solution(
            bits=np.array([1, 0, 1, 0], dtype=bool),
            variables=[np.array([1, 0], dtype=bool), np.array([1, 0], dtype=bool)]
        )
        
        # Create the crossover instance
        crossover = SPXCrossover(1.0)
        
        # Mock the instance's _rng
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.5  # Will perform crossover
        mock_rng.integers.return_value = 2  # Crossover after 2nd bit
        
        # Replace the instance's _rng with our mock
        with patch.object(crossover, '_rng', mock_rng):
            offspring = crossover.execute([parent1, parent2])
            
            # Should return two offspring
            assert len(offspring) == 2
            # Verify the crossover happened correctly at bit position 2
            expected_offspring1 = np.array([0, 1, 1, 0], dtype=bool)
            expected_offspring2 = np.array([1, 0, 0, 1], dtype=bool)
            assert np.array_equal(offspring[0].bits, expected_offspring1)
            assert np.array_equal(offspring[1].bits, expected_offspring2)

    def test_given_probability_zero_when_executing_then_never_performs_crossover(self):
        """Test that crossover is never performed when random value is not less than probability."""
        parent1 = self.create_mock_solution(
            bits=np.array([0, 1, 0, 1], dtype=bool),
            variables=[np.array([0, 1], dtype=bool), np.array([0, 1], dtype=bool)]
        )
        parent2 = self.create_mock_solution(
            bits=np.array([1, 0, 1, 0], dtype=bool),
            variables=[np.array([1, 0], dtype=bool), np.array([1, 0], dtype=bool)]
        )
        
        # Create the crossover instance
        crossover = SPXCrossover(0.0)  # 0% probability
        
        # Mock the instance's _rng
        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.0  # Will not perform crossover
        
        # Replace the instance's _rng with our mock
        with patch.object(crossover, '_rng', mock_rng):
            offspring = crossover.execute([parent1, parent2])
            
            # Should return two offspring identical to parents
            assert len(offspring) == 2
            assert np.array_equal(offspring[0].bits, parent1.bits)
            assert np.array_equal(offspring[1].bits, parent2.bits)

    def test_given_single_bit_solution_when_executing_then_no_crossover_happens(self):
        """Test that crossover doesn't happen with single-bit solutions."""
        parent1 = self.create_mock_solution(
            bits=np.array([True], dtype=bool),
            variables=[np.array([True], dtype=bool)]
        )
        parent2 = self.create_mock_solution(
            bits=np.array([False], dtype=bool),
            variables=[np.array([False], dtype=bool)]
        )
        
        # Create the crossover instance
        crossover = SPXCrossover(1.0)
        
        # Mock the instance's _rng
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.5  # Would perform crossover if not for single bit
        
        # Replace the instance's _rng with our mock
        with patch.object(crossover, '_rng', mock_rng):
            offspring = crossover.execute([parent1, parent2])
            
            # With single bit, no crossover should happen (num_bits <= 1)
            assert len(offspring) == 2
            # Offspring should be copies of parents
            assert np.array_equal(offspring[0].bits, parent1.bits)
            assert np.array_equal(offspring[1].bits, parent2.bits)
            # Verify _rng.integers was not called (since we don't perform crossover on single bit)
            mock_rng.integers.assert_not_called()


class TestBitFlipMutation:
    def test_given_valid_probability_when_initializing_then_probability_is_set_correctly(self):
        """Test initialization with valid probability values."""
        # Test with minimum probability
        # Arrange
        min_probability = 0.0
        
        # Act
        mutation = BitFlipMutation(min_probability)
        
        # Assert
        assert mutation.probability == min_probability
        
        # Test with maximum probability
        # Arrange
        max_probability = 1.0
        
        # Act
        mutation = BitFlipMutation(max_probability)
        
        # Assert
        assert mutation.probability == max_probability
        
        # Test with mid-range probability
        # Arrange
        mid_probability = 0.5
        
        # Act
        mutation = BitFlipMutation(mid_probability)
        
        # Assert
        assert mutation.probability == mid_probability

    def test_given_negative_probability_when_initializing_then_raises_value_error(self):
        """Test initialization with negative probability."""
        # Arrange
        negative_probability = -0.1
        
        # Act & Assert
        with pytest.raises(ValueError, match="Probability must be in range \[0.0, 1.0\], got -0.1"):
            BitFlipMutation(negative_probability)
    
    def test_given_probability_above_one_when_initializing_then_raises_value_error(self):
        """Test initialization with probability greater than 1.0."""
        # Arrange
        high_probability = 1.1
        
        # Act & Assert
        with pytest.raises(ValueError, match="Probability must be in range \[0.0, 1.0\], got 1.1"):
            BitFlipMutation(high_probability)

    def test_given_bit_flip_mutation_when_getting_name_then_returns_correct_string(self):
        """Test the get_name method returns the correct string."""
        # Arrange
        mutation = BitFlipMutation(0.5)
        expected_name = "Bit flip mutation"
        
        # Act
        name = mutation.get_name()
        
        # Assert
        assert name == expected_name

    def test_given_valid_binary_solution_when_executing_mutation_then_bits_are_flipped_according_to_probability(self):
        """Test mutation with a valid binary solution."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 10
        solution.bits = np.array([False] * 10)
        mutation_probability = 0.5
        
        # Patch random to always return 0.1 (so with probability 0.5, all bits should flip)
        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = np.array([0.1] * 10)
            
            # Act
            mutation = BitFlipMutation(mutation_probability)
            result = mutation.execute(solution)
            
            # Assert
            assert result is solution  # Should return the same instance
            assert np.all(solution.bits == True)  # All bits should be flipped

    def test_given_zero_variables_when_executing_mutation_then_raises_value_error(self):
        """Test mutation with a solution that has zero variables."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 0
        solution.bits = np.array([])
        mutation = BitFlipMutation(0.5)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Solution must have at least one variable"):
            mutation.execute(solution)

    def test_given_non_binary_solution_when_executing_mutation_then_raises_type_error(self):
        """Test mutation with an invalid solution type."""
        # Arrange
        class OtherSolution:
            pass
            
        mutation = BitFlipMutation(0.5)
        invalid_solution = OtherSolution()
        
        # Act & Assert
        with pytest.raises(TypeError, match="Expected BinarySolution, got OtherSolution"):
            mutation.execute(invalid_solution)

    def test_given_solution_without_bits_attribute_when_executing_mutation_then_raises_attribute_error(self):
        """Test mutation with a solution missing the bits attribute."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 10
        del solution.bits  # Remove the bits attribute
        mutation = BitFlipMutation(0.5)
        
        # Act & Assert
        with pytest.raises(AttributeError, match="must have a 'bits' attribute"):
            mutation.execute(solution)

    def test_given_solution_with_non_numpy_bits_when_executing_mutation_then_raises_attribute_error(self):
        """Test mutation with a solution where bits is not a numpy array."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 10
        solution.bits = [False] * 10  # Not a numpy array
        mutation = BitFlipMutation(0.5)
        
        # Act & Assert
        with pytest.raises(AttributeError, match="must have a 'bits' attribute of type numpy.ndarray"):
            mutation.execute(solution)

    def test_given_zero_mutation_probability_when_executing_mutation_then_no_bits_flip(self):
        """Test that with zero probability, no bits are flipped."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 1000
        solution.bits = np.array([False] * 1000)
        zero_probability = 0.0
        
        # Act
        mutation = BitFlipMutation(zero_probability)
        result = mutation.execute(solution)
        
        # Assert
        assert np.sum(result.bits) == 0
    
    def test_given_full_mutation_probability_when_executing_mutation_then_all_bits_flip(self):
        """Test that with probability 1.0, all bits are flipped."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 1000
        solution.bits = np.array([False] * 1000)
        full_probability = 1.0
        
        # Act
        mutation = BitFlipMutation(full_probability)
        result = mutation.execute(solution)
        
        # Assert
        assert np.all(result.bits == True)

    def test_given_large_solution_when_executing_mutation_then_mutation_rate_matches_probability(self):
        """Test that the actual mutation rate matches the specified probability."""
        # Arrange
        solution = MagicMock(spec=BinarySolution)
        solution.number_of_variables = 10000  # Large sample size
        solution.bits = np.array([False] * 10000)
        target_probability = 0.3
        
        # Act
        mutation = BitFlipMutation(target_probability)
        result = mutation.execute(solution)
        
        # Assert
        actual_mutation_rate = np.sum(result.bits) / len(result.bits)
        assert 0.28 <= actual_mutation_rate <= 0.32  # Allow some statistical variation
