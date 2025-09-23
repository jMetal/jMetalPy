import pytest
from unittest.mock import Mock, MagicMock, call
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.metaopt.component.solutions_creation import DefaultSolutionsCreation, SolutionsCreation

class TestDefaultSolutionsCreation:
    """Tests for the DefaultSolutionsCreation implementation."""
    
    @pytest.fixture
    def problem(self):
        problem = Mock(spec=Problem)
        problem.create_solution.side_effect = lambda: Mock(spec=Solution)
        return problem
    
    @pytest.fixture
    def creator(self, problem):
        return DefaultSolutionsCreation(problem=problem, number_of_solutions_to_create=10)
    
    class TestInitialization:
        """Tests for DefaultSolutionsCreation initialization."""
        
        def test_raises_error_when_problem_is_none(self):
            """Should raise ValueError when problem is None."""
            with pytest.raises(ValueError, match="Problem cannot be None"):
                DefaultSolutionsCreation(problem=None, number_of_solutions_to_create=1)
        
        @pytest.mark.parametrize("invalid_size", [0, -1, "10", 3.14, None])
        def test_raises_error_with_invalid_solution_count(self, problem, invalid_size):
            """Should raise ValueError with invalid solution counts."""
            with pytest.raises(ValueError, match="must be a positive integer"):
                DefaultSolutionsCreation(problem=problem, number_of_solutions_to_create=invalid_size)
    
    class TestCreateMethod:
        """Tests for the create() method."""
        
        def test_returns_correct_number_of_solutions(self, creator, problem):
            """Should return the specified number of solutions."""
            # When
            solutions = creator.create()
            
            # Then
            assert len(solutions) == 10
            assert problem.create_solution.call_count == 10
        
        def test_returns_unique_solution_instances(self, creator, problem):
            """Should return unique solution instances."""
            # Given
            problem.create_solution.side_effect = [Mock(spec=Solution) for _ in range(10)]
            
            # When
            solutions = creator.create()
            
            # Then
            assert len({id(sol) for sol in solutions}) == 10