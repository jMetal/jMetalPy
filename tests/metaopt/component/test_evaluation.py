import pytest
from unittest.mock import Mock, call, ANY
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.metaopt.component.evaluation import SequentialEvaluation


class TestSequentialEvaluation:
    """Tests for the SequentialEvaluation implementation."""
    
    @pytest.fixture
    def problem(self):
        """Create a mock problem that marks solutions as evaluated."""
        problem = Mock(spec=Problem)
        problem.evaluate.side_effect = lambda s: setattr(s, 'evaluated', True)
        return problem
    
    @pytest.fixture
    def solution(self):
        """Create a single mock solution."""
        return Mock(spec=Solution)
    
    @pytest.fixture
    def solutions(self):
        """Create multiple mock solutions."""
        def _create_solutions(count: int) -> list[Mock]:
            return [Mock(spec=Solution) for _ in range(count)]
        return _create_solutions
    
    @pytest.fixture
    def evaluator(self, problem):
        """Create a SequentialEvaluation instance with a mock problem."""
        return SequentialEvaluation(problem)
    
    class TestInitialization:
        """Tests for SequentialEvaluation initialization."""
        
        def test_raises_error_when_problem_is_none(self):
            """Should raise ValueError when problem is None."""
            with pytest.raises(ValueError, match="Problem instance cannot be None"):
                SequentialEvaluation(problem=None)  # type: ignore
    
    class TestEvaluateMethod:
        """Tests for the evaluate() method."""
        
        def test_returns_empty_list_when_input_is_empty(self, evaluator, problem):
            """Should return an empty list when input is empty."""
            # When
            result = evaluator.evaluate([])
            
            # Then
            assert result == []
            assert evaluator.computed_evaluations() == 0
            problem.evaluate.assert_not_called()
        
        def test_evaluates_single_solution(self, evaluator, problem, solution):
            """Should evaluate a single solution and update evaluation count."""
            # Given
            solutions = [solution]
            
            # When
            result = evaluator.evaluate(solutions)
            
            # Then
            assert result is solutions
            problem.evaluate.assert_called_once_with(solution)
            assert evaluator.computed_evaluations() == 1
            assert hasattr(solution, 'evaluated')
        
        def test_evaluates_multiple_solutions_in_order(self, evaluator, problem, solutions):
            """Should evaluate multiple solutions in the order provided."""
            # Given
            solution_list = solutions(3)
            
            # When
            result = evaluator.evaluate(solution_list)
            
            # Then
            assert result is solution_list
            assert evaluator.computed_evaluations() == 3
            expected_calls = [call(sol) for sol in solution_list]
            problem.evaluate.assert_has_calls(expected_calls, any_order=False)
            
            for solution in solution_list:
                assert hasattr(solution, 'evaluated')
    
    class TestComputedEvaluations:
        """Tests for the computed_evaluations() method."""
        
        def test_counts_accumulate_across_multiple_calls(self, evaluator, solutions):
            """Should accumulate evaluation counts across multiple evaluate() calls."""
            # Given
            solution_batch = solutions(2)
            
            # When - First evaluation
            evaluator.evaluate(solution_batch[:1])
            
            # Then
            assert evaluator.computed_evaluations() == 1
            
            # When - Second evaluation
            evaluator.evaluate(solution_batch[1:])
            
            # Then
            assert evaluator.computed_evaluations() == 2