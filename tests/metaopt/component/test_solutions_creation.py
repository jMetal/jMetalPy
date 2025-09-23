import pytest
from typing import List, TypeVar
from unittest.mock import Mock, MagicMock, call
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.metaopt.component.solutions_creation import DefaultSolutionsCreation, SolutionsCreation

# Type variable for solutions
S = TypeVar('S', bound=Solution)

# Fixture for a mock problem
@pytest.fixture
def mock_problem():
    problem = Mock(spec=Problem)
    problem.create_solution.side_effect = lambda: Mock(spec=Solution)
    return problem

# Fixture for DefaultSolutionsCreation
@pytest.fixture
def default_creator(mock_problem):
    return DefaultSolutionsCreation(problem=mock_problem, number_of_solutions_to_create=10)

class TestDefaultSolutionsCreation:
    """Tests for the DefaultSolutionsCreation implementation."""
    
    def test_initialization_with_none_problem(self):
        """Test that initializing with None problem raises ValueError."""
        # Arrange
        problem = None
        number_of_solutions = 10

        # Act & Assert
        with pytest.raises(ValueError, match="Problem cannot be None"):
            DefaultSolutionsCreation(problem=problem, number_of_solutions_to_create=number_of_solutions)
    
    @pytest.mark.parametrize("invalid_size", [0, -1, "10", 3.14, None])
    def test_initialization_with_invalid_number_of_solutions(self, mock_problem, invalid_size):
        """Test that initializing with invalid number of solutions raises ValueError."""
        # Arrange
        problem = mock_problem
        
        # Act & Assert
        with pytest.raises(ValueError, match="Number of solutions to create must be a positive integer"):
            DefaultSolutionsCreation(problem=problem, number_of_solutions_to_create=invalid_size)
    
    def test_create_returns_correct_number_of_solutions(self, default_creator, mock_problem):
        """Test that create() returns the correct number of solutions."""
        # Act
        solutions = default_creator.create()
        
        # Assert
        assert len(solutions) == 10
        assert mock_problem.create_solution.call_count == 10
    
    def test_create_returns_unique_solution_instances(self, default_creator, mock_problem):
        """Test that create() returns unique solution instances."""
        # Arrange
        expected_count = 10
        mock_problem.create_solution.side_effect = [Mock(spec=Solution) for _ in range(expected_count)]
        
        # Act
        solutions = default_creator.create()
        
        # Assert
        solution_ids = [id(sol) for sol in solutions]
        unique_solution_ids = set(solution_ids)
        assert len(solutions) == expected_count
        assert len(solution_ids) == len(unique_solution_ids)
    
    def test_create_uses_problem_create_solution(self, default_creator, mock_problem):
        """Test that create() uses the problem's create_solution method."""
        # Act
        solutions = default_creator.create()
        
        # Assert
        assert mock_problem.create_solution.call_count == 10
        for solution in solutions:
            assert isinstance(solution, Mock)
    
    @pytest.mark.parametrize("size", [1, 5, 100])
    def test_create_with_different_sizes(self, mock_problem, size):
        """Test create() with different population sizes."""
        # Arrange
        mock_problem.create_solution.reset_mock()
        creator = DefaultSolutionsCreation(problem=mock_problem, number_of_solutions_to_create=size)
        
        # Act
        solutions = creator.create()
        
        # Assert
        assert len(solutions) == size
        assert mock_problem.create_solution.call_count == size
    
    def test_solutions_creation_implements_interface(self, default_creator):
        """Test that DefaultSolutionsCreation properly implements the SolutionsCreation interface."""
        # Arrange
        creator = default_creator
        
        # Act & Assert
        assert isinstance(creator, SolutionsCreation), "Should be an instance of SolutionsCreation"
        assert hasattr(creator, 'create'), "Should have a 'create' method"
        assert callable(getattr(creator, 'create')), "'create' should be callable"
