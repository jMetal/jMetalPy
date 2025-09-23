import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, call, ANY
from typing import List
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from jmetal.core.solution import Solution
from src.jmetal.metaopt.component.selection import RandomSelection, Selection


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
