import pytest
from typing import List, TypeVar
from unittest.mock import Mock, call
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.metaopt.component.evaluation import SequentialEvaluation

# Type variable for solutions
S = TypeVar('S', bound=Solution)

# Fixture for a mock problem
@pytest.fixture
def mock_problem():
    problem = Mock(spec=Problem)
    problem.evaluate.side_effect = lambda s: setattr(s, 'evaluated', True)
    return problem

# Fixture for a mock solution
@pytest.fixture
def mock_solution():
    return Mock(spec=Solution)

# Fixture for multiple mock solutions
@pytest.fixture
def mock_solutions():
    def _create_solutions(n: int) -> List[Mock]:
        return [Mock(spec=Solution) for _ in range(n)]
    return _create_solutions

# Fixture for SequentialEvaluation
@pytest.fixture
def sequential_evaluator(mock_problem):
    return SequentialEvaluation(mock_problem)

class TestSequentialEvaluation:
    """Tests for the SequentialEvaluation implementation."""
    
    def test_initialization_with_none_problem(self):
        """Test that initializing with None problem raises ValueError."""
        # Arrange
        problem = None

        # Act & Assert
        with pytest.raises(ValueError):
            SequentialEvaluation(problem)  # type: ignore
    
    def test_evaluate_empty_list(self, sequential_evaluator, mock_problem):
        """Test evaluating an empty solution list."""
        # Arrange
        empty_list = []

        # Act
        result = sequential_evaluator.evaluate(empty_list)

        # Assert
        assert result == []
        assert sequential_evaluator.computed_evaluations() == 0
        mock_problem.evaluate.assert_not_called()
    
    def test_evaluate_single_solution(self, sequential_evaluator, mock_problem, mock_solution):
        """Test evaluating a single solution."""
        # Arrange
        solutions = [mock_solution]

        # Act
        result = sequential_evaluator.evaluate(solutions)

        # Assert
        assert result is solutions
        mock_problem.evaluate.assert_called_once_with(mock_solution)
        assert sequential_evaluator.computed_evaluations() == 1
    
    def test_evaluate_multiple_solutions(self, sequential_evaluator, mock_problem, mock_solutions):
        """Test evaluating multiple solutions."""
        # Arrange
        solutions = mock_solutions(3)

        # Act
        result = sequential_evaluator.evaluate(solutions)

        # Assert
        assert result is solutions
        assert sequential_evaluator.computed_evaluations() == 3
        expected_calls = [call(sol) for sol in solutions]
        mock_problem.evaluate.assert_has_calls(expected_calls, any_order=False)
    
    def test_computed_evaluations_accumulation(self, sequential_evaluator, mock_solutions):
        """Test that evaluation counts accumulate across multiple calls."""
        # Arrange
        solutions = mock_solutions(2)

        # Act - First evaluation
        sequential_evaluator.evaluate(solutions[:1])

        # Assert
        assert sequential_evaluator.computed_evaluations() == 1

        # Act - Second evaluation
        sequential_evaluator.evaluate(solutions[1:])

        # Assert
        assert sequential_evaluator.computed_evaluations() == 2