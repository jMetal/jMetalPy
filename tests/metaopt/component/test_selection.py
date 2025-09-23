import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, call, ANY, patch
from typing import List
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator, DominanceComparator
from src.jmetal.metaopt.component.selection import RandomSelection, Selection, NaryTournamentSelection


class TestRandomSelection:
    """Tests for the RandomSelection implementation."""

    @pytest.fixture
    def solution(self) -> Solution:
        """Create a mock solution."""
        return Mock(spec=Solution)

    @pytest.fixture
    def solutions(self, solution: Solution) -> List[Solution]:
        """Create a list of mock solutions."""
        def _create_solutions(count: int) -> List[Solution]:
            return [Mock(spec=Solution) for _ in range(count)]
        return _create_solutions

    class TestInitialization:
        """Tests for RandomSelection initialization."""

        def test_raises_error_when_number_of_solutions_is_not_positive(self):
            """Should raise ValueError when number_of_solutions_to_select is not positive."""
            # Test with zero
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                RandomSelection(number_of_solutions_to_select=0)
            
            # Test with negative number
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                RandomSelection(number_of_solutions_to_select=-5)
            
            # Test with non-integer
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                RandomSelection(number_of_solutions_to_select=3.14)  # type: ignore

        def test_initializes_correctly_with_valid_parameters(self):
            """Should initialize with default parameters correctly."""
            # When
            selector = RandomSelection(number_of_solutions_to_select=5)
            
            # Then
            assert selector.number_of_solutions_to_select == 5
            assert selector.with_replacement is False

        def test_initializes_with_replacement_correctly(self):
            """Should initialize with replacement set to True when specified."""
            # When
            selector = RandomSelection(number_of_solutions_to_select=3, with_replacement=True)
            
            # Then
            assert selector.with_replacement is True

    class TestSelectMethod:
        """Tests for the select() method."""

        def test_raises_error_when_solution_list_is_empty(self, solutions):
            """Should raise ValueError when solution_list is empty."""
            # Given
            selector = RandomSelection(number_of_solutions_to_select=1)
            
            # When / Then
            with pytest.raises(ValueError, match="Cannot select from an empty or None solution list"):
                selector.select([])

        def test_raises_error_when_solution_list_is_none(self, solutions):
            """Should raise ValueError when solution_list is None."""
            # Given
            selector = RandomSelection(number_of_solutions_to_select=1)
            
            # When / Then
            with pytest.raises(ValueError, match="Cannot select from an empty or None solution list"):
                selector.select(None)  # type: ignore

        def test_selects_correct_number_of_solutions_without_replacement(self, solutions):
            """Should return the correct number of unique solutions when not using replacement."""
            # Given
            solution_list = solutions(5)
            selector = RandomSelection(number_of_solutions_to_select=3, with_replacement=False)
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert len(selected) == 3
            assert len(set(id(s) for s in selected)) == 3  # All solutions are unique

        def test_selects_correct_number_of_solutions_with_replacement(self, solutions):
            """Should return the correct number of solutions when using replacement."""
            # Given
            solution_list = solutions(2)  # Only 2 solutions available
            selector = RandomSelection(number_of_solutions_to_select=3, with_replacement=True)
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert len(selected) == 3
            # With replacement, we can have duplicates, so we don't check for uniqueness

        def test_raises_error_when_selecting_too_many_without_replacement(self, solutions):
            """Should raise ValueError when trying to select more solutions than available without replacement."""
            # Given
            solution_list = solutions(3)
            selector = RandomSelection(number_of_solutions_to_select=4, with_replacement=False)
            
            # When / Then
            with pytest.raises(
                ValueError, 
                match=r"Cannot select 4 solutions without replacement from a list of 3 solutions"
            ):
                selector.select(solution_list)

        def test_returns_all_solutions_when_selecting_all(self, solutions):
            """Should return all solutions when selecting exactly the number available."""
            # Given
            solution_list = solutions(4)
            selector = RandomSelection(number_of_solutions_to_select=4, with_replacement=False)
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert len(selected) == 4
            assert set(id(s) for s in selected) == set(id(s) for s in solution_list)

    class TestRandomness:
        """Tests related to the randomness behavior of the selection."""

        def test_selection_is_random(self, solutions, monkeypatch):
            """Should select solutions randomly."""
            # Given
            solution_list = solutions(10)
            selector = RandomSelection(number_of_solutions_to_select=5, with_replacement=False)
            
            # Mock random.sample to return the first 5 elements
            mock_sample = MagicMock(return_value=solution_list[:5])
            monkeypatch.setattr(random, 'sample', mock_sample)
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert mock_sample.call_count == 1
            args, kwargs = mock_sample.call_args
            assert args[0] == solution_list  # First argument is the population
            assert kwargs.get('k') == 5 or (len(args) > 1 and args[1] == 5)  # k can be positional or keyword arg
            assert selected == solution_list[:5]

        def test_selection_with_replacement_uses_choices(self, solutions, monkeypatch):
            """Should use random.choices when with_replacement is True."""
            # Given
            solution_list = solutions(3)
            selector = RandomSelection(number_of_solutions_to_select=5, with_replacement=True)
            
            # We'll create a list of 5 elements by repeating the first solution
            expected_result = [solution_list[0]] * 5
            mock_choices = MagicMock(return_value=expected_result)
            monkeypatch.setattr(random, 'choices', mock_choices)
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert mock_choices.call_count == 1
            args, kwargs = mock_choices.call_args
            assert args[0] == solution_list  # First argument is the population
            assert kwargs.get('k') == 5 or (len(args) > 1 and args[1] == 5)  # k can be positional or keyword arg
            assert 'weights' not in kwargs or kwargs['weights'] is None  # No weights should be provided
            assert len(selected) == 5
            assert selected == expected_result

    def test_implements_selection_interface(self):
        """Test that RandomSelection properly implements the Selection interface."""
        # Given
        selector = RandomSelection(number_of_solutions_to_select=1)
        
        # Then
        assert isinstance(selector, Selection), "RandomSelection should implement the Selection interface"
        assert hasattr(selector, 'select'), "RandomSelection should have a 'select' method"
        assert callable(selector.select), "'select' should be callable"

    def test_string_representation(self):
        """Test the string representation of RandomSelection."""
        # Given
        selector = RandomSelection(number_of_solutions_to_select=3, with_replacement=True)
        
        # When
        str_repr = str(selector)
        
        # Then
        assert "RandomSelection(" in str_repr
        assert "number_of_solutions_to_select=3" in str_repr
        assert "with_replacement=True" in str_repr


class TestNaryTournamentSelection:
    """Tests for the NaryTournamentSelection implementation."""

    @pytest.fixture
    def solution(self) -> Solution:
        """Create a mock solution."""
        solution = Mock(spec=Solution)
        solution.objectives = [1.0]  # Default objective value
        return solution

    @pytest.fixture
    def solutions(self, solution: Solution) -> List[Solution]:
        """Create a list of mock solutions with different objective values."""
        def _create_solutions(count: int) -> List[Solution]:
            solutions = []
            for i in range(count):
                s = Mock(spec=Solution)
                s.objectives = [float(i)]  # Each solution has a different objective value
                solutions.append(s)
            return solutions
        return _create_solutions

    @pytest.fixture
    def mock_comparator(self) -> Comparator:
        """Create a mock comparator that prefers solutions with lower objective values."""
        class MockComparator(Comparator):
            def compare(self, solution1: Solution, solution2: Solution) -> int:
                if solution1.objectives[0] < solution2.objectives[0]:
                    return -1
                elif solution1.objectives[0] > solution2.objectives[0]:
                    return 1
                return 0
        return MockComparator()

    class TestInitialization:
        """Tests for NaryTournamentSelection initialization."""

        def test_raises_error_when_number_of_solutions_is_not_positive(self):
            """Should raise ValueError when number_of_solutions_to_select is not positive."""
            # Test with zero
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                NaryTournamentSelection(number_of_solutions_to_select=0)
            
            # Test with negative number
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                NaryTournamentSelection(number_of_solutions_to_select=-5)
            
            # Test with non-integer
            with pytest.raises(ValueError, match="Number of solutions to select must be a positive integer"):
                NaryTournamentSelection(number_of_solutions_to_select=3.14)  # type: ignore

        def test_raises_error_when_tournament_size_is_out_of_bounds(self):
            """Should raise ValueError when tournament_size is not between 2 and 10."""
            # Test with 1
            with pytest.raises(ValueError, match="Tournament size must be between 2 and 10"):
                NaryTournamentSelection(number_of_solutions_to_select=2, tournament_size=1)
            
            # Test with 11
            with pytest.raises(ValueError, match="Tournament size must be between 2 and 10"):
                NaryTournamentSelection(number_of_solutions_to_select=2, tournament_size=11)

        def test_initializes_correctly_with_default_parameters(self):
            """Should initialize with default parameters correctly."""
            # When
            with patch('jmetal.util.comparator.DominanceComparator') as mock_comparator_class:
                mock_comparator = mock_comparator_class.return_value
                selector = NaryTournamentSelection(number_of_solutions_to_select=3)
                
                # Then
                assert selector.number_of_solutions_to_select == 3
                assert selector.tournament_size == 2  # Default tournament size
                assert isinstance(selector.comparator, Comparator)

        def test_initializes_with_custom_comparator(self, mock_comparator):
            """Should initialize with a custom comparator when provided."""
            # When
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=3,
                tournament_size=3,
                comparator=mock_comparator
            )
            
            # Then
            assert selector.comparator is mock_comparator

    class TestSelectMethod:
        """Tests for the select() method."""

        def test_raises_error_when_solution_list_is_empty(self, solutions, mock_comparator):
            """Should raise ValueError when solution_list is empty."""
            # Given
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=1, 
                comparator=mock_comparator
            )
            
            # When / Then
            with pytest.raises(ValueError, match="Cannot select from an empty or None solution list"):
                selector.select([])

        def test_raises_error_when_solution_list_is_none(self, solutions, mock_comparator):
            """Should raise ValueError when solution_list is None."""
            # Given
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=1, 
                comparator=mock_comparator
            )
            
            # When / Then
            with pytest.raises(ValueError, match="Cannot select from an empty or None solution list"):
                selector.select(None)  # type: ignore

        def test_raises_error_when_tournament_size_larger_than_population(self, solutions, mock_comparator):
            """Should raise ValueError when tournament size is larger than population size."""
            # Given
            solution_list = solutions(3)  # Only 3 solutions
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=2,
                tournament_size=4,  # Larger than population
                comparator=mock_comparator
            )
            
            # When / Then
            with pytest.raises(
                ValueError, 
                match=r"Tournament size \(4\) cannot be larger than the number of available solutions \(3\)"
            ):
                selector.select(solution_list)

        def test_selects_correct_number_of_solutions(self, solutions, mock_comparator):
            """Should return the correct number of solutions."""
            # Given
            solution_list = solutions(10)
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=4,
                tournament_size=3,
                comparator=mock_comparator
            )
            
            # Mock random.sample to control tournament participants
            with patch('random.sample') as mock_sample:
                # Make the mock return the first 3 solutions for each tournament
                mock_sample.side_effect = [solution_list[:3]] * 4
                
                # When
                selected = selector.select(solution_list)
                
                # Then
                assert len(selected) == 4
                # Since we mocked random.sample to always return the first 3 solutions,
                # and our mock_comparator prefers solutions with lower objective values,
                # the first solution (with objectives=[0.0]) should always be selected
                assert all(s.objectives[0] == 0.0 for s in selected)

        def test_binary_tournament_default_behavior(self, solutions, mock_comparator):
            """Should work correctly with default binary tournament (N=2)."""
            # Given
            solution_list = solutions(5)  # Solutions with objectives [0.0, 1.0, 2.0, 3.0, 4.0]
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=3,
                tournament_size=2,  # Binary tournament
                comparator=mock_comparator
            )
            
            # Mock random.sample to control tournament participants
            with patch('random.sample') as mock_sample:
                # Simulate 3 binary tournaments, each time choosing solutions 0 and 1
                mock_sample.side_effect = [
                    [solution_list[0], solution_list[1]],  # First tournament: 0 vs 1 -> 0 wins
                    [solution_list[0], solution_list[2]],  # Second tournament: 0 vs 2 -> 0 wins
                    [solution_list[1], solution_list[2]]   # Third tournament: 1 vs 2 -> 1 wins
                ]
                
                # When
                selected = selector.select(solution_list)
                
                # Then
                assert len(selected) == 3
                # First two should be solution 0 (best in its tournaments)
                assert selected[0].objectives[0] == 0.0
                assert selected[1].objectives[0] == 0.0
                # Third should be solution 1 (best in its tournament)
                assert selected[2].objectives[0] == 1.0

        def test_uses_provided_comparator(self, solutions, mock_comparator):
            """Should use the provided comparator to determine tournament winners."""
            # Given
            solution_list = solutions(3)  # Solutions with objectives [0.0, 1.0, 2.0]
            
            # Create a selector with a mock comparator
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=1,
                tournament_size=2,
                comparator=mock_comparator
            )
            
            # Mock random.sample to control tournament participants
            with patch('random.sample') as mock_sample:
                # Simulate a tournament between solutions 1 and 2 (1.0 vs 2.0)
                mock_sample.return_value = [solution_list[1], solution_list[2]]
                
                # When
                selected = selector.select(solution_list)
                
                # Then
                # The mock_comparator prefers lower objective values, so solution 1 (1.0) should win
                assert len(selected) == 1
                assert selected[0].objectives[0] == 1.0

        def test_handles_single_solution_population(self, solutions, mock_comparator):
            """Should handle case when population size equals tournament size."""
            # Given
            solution_list = solutions(2)  # Need at least 2 solutions for minimum tournament size
            selector = NaryTournamentSelection(
                number_of_solutions_to_select=1,
                tournament_size=2,  # Minimum tournament size is 2
                comparator=mock_comparator
            )
            
            # When
            selected = selector.select(solution_list)
            
            # Then
            assert len(selected) == 1
            assert selected[0] in solution_list
    
    def test_string_representation(self, mock_comparator):
        """Test the string representation of NaryTournamentSelection."""
        # Given
        selector = NaryTournamentSelection(
            number_of_solutions_to_select=3,
            tournament_size=4,
            comparator=mock_comparator
        )
        
        # When
        str_repr = str(selector)
        
        # Then
        assert "NTournamentSelection(" in str_repr
        assert "number_of_solutions_to_select=3" in str_repr
        assert "tournament_size=4" in str_repr
