"""Tests for selection operators."""
from unittest.mock import patch
import pytest

from jmetal.core.solution import Solution, FloatSolution
from jmetal.operator.selection import (
    BinaryTournamentSelection,
    BestSolutionSelection,
    RandomSelection,
    DifferentialEvolutionSelection,
    NaryRandomSolutionSelection,
    RankingAndCrowdingDistanceSelection,
    TournamentSelection,
)

# Dummy solution class for testing
class DummySolution(Solution):
    """Dummy solution class for testing purposes."""
    def __init__(self, objectives, variables=None):
        if variables is None:
            variables = [0.0] * len(objectives)
        super().__init__(len(variables), len(objectives))
        self.objectives = objectives
        self.variables = variables
        self.attributes = {}
        self.number_of_constraints = 0
        self.constraints = []
        self.constraint_violation = 0.0

class TestBinaryTournamentSelection:
    """Tests for BinaryTournamentSelection operator."""
    
    def test_should_initialize_with_default_parameters(self):
        selector = BinaryTournamentSelection()
        assert selector is not None

    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = BinaryTournamentSelection()
        with pytest.raises(Exception):
            selector.execute(invalid_input)

    def test_should_return_single_solution_when_list_has_one_solution(self, float_solution_factory):
        solution = float_solution_factory([1.0, 2.0])
        selector = BinaryTournamentSelection()
        
        result = selector.execute([solution])
        
        assert result == solution

    @patch('random.sample', return_value=[0, 1])
    def test_should_select_between_two_non_dominated_solutions(self, _, float_solution_factory):
        solution1 = float_solution_factory([1.0, 1.0])
        solution2 = float_solution_factory([0.0, 2.0])
        selector = BinaryTournamentSelection()
        
        selection = selector.execute([solution1, solution2])
        
        assert selection in [solution1, solution2]

    @patch('random.sample')
    def test_should_work_with_five_solutions(self, mock_sample, float_solution_factory):
        solutions = [
            float_solution_factory([1.0, 1.0]),
            float_solution_factory([0.0, 2.0]),
            float_solution_factory([0.5, 1.5]),
            float_solution_factory([0.0, 0.0]),
            float_solution_factory([1.0, 0.0])
        ]
        mock_sample.side_effect = lambda x, k: [0, 1]  # Always select first two solutions
        
        selector = BinaryTournamentSelection()
        selection = selector.execute(solutions)
        
        assert selection in solutions

class TestTournamentSelection:
    """Tests for TournamentSelection operator (k-ary tournament)."""
    
    def test_should_initialize_with_default_parameters(self):
        selector = TournamentSelection()
        assert selector is not None
        assert selector.tournament_size == 2
    
    def test_should_initialize_with_custom_tournament_size(self):
        selector = TournamentSelection(tournament_size=5)
        assert selector.tournament_size == 5
    
    def test_should_raise_exception_for_tournament_size_less_than_2(self):
        with pytest.raises(ValueError) as exc_info:
            TournamentSelection(tournament_size=1)
        assert "Tournament size must be at least 2" in str(exc_info.value)
    
    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = TournamentSelection()
        with pytest.raises(Exception):
            selector.execute(invalid_input)
    
    def test_should_return_single_solution_when_list_has_one_solution(self, float_solution_factory):
        solution = float_solution_factory([1.0, 2.0])
        selector = TournamentSelection(tournament_size=3)
        
        result = selector.execute([solution])
        
        assert result == solution
    
    def test_should_adjust_tournament_size_if_population_is_smaller(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(3)]
        selector = TournamentSelection(tournament_size=10)  # Larger than population
        
        # Should not raise an exception
        result = selector.execute(solutions)
        assert result in solutions
    
    @patch('random.sample')
    def test_should_select_best_from_tournament(self, mock_sample, float_solution_factory):
        # Create solutions where solution at index 3 dominates others
        solutions = [
            float_solution_factory([5.0, 5.0]),  # 0 - dominated
            float_solution_factory([4.0, 4.0]),  # 1 - dominated
            float_solution_factory([3.0, 3.0]),  # 2 - dominated
            float_solution_factory([1.0, 1.0]),  # 3 - dominates all others
            float_solution_factory([6.0, 6.0]),  # 4 - dominated
        ]
        # Mock sample to return indices 0, 1, 3 (tournament participants)
        mock_sample.return_value = [0, 1, 3]
        
        selector = TournamentSelection(tournament_size=3)
        result = selector.execute(solutions)
        
        # Should select solution 3 as it dominates 0 and 1
        assert result == solutions[3]
    
    def test_should_return_solution_from_population(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(10)]
        selector = TournamentSelection(tournament_size=4)
        
        for _ in range(20):  # Run multiple times to test randomness
            result = selector.execute(solutions)
            assert result in solutions
    
    def test_should_have_correct_name(self):
        selector = TournamentSelection(tournament_size=5)
        assert selector.get_name() == "Tournament selection (k=5)"
    
    def test_should_work_with_custom_comparator(self, float_solution_factory):
        from jmetal.util.comparator import DominanceComparator
        
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(5)]
        selector = TournamentSelection(tournament_size=3, comparator=DominanceComparator())
        
        result = selector.execute(solutions)
        assert result in solutions


class TestBestSolutionSelection:
    """Tests for BestSolutionSelection operator."""
    
    def test_should_initialize_correctly(self):
        assert BestSolutionSelection() is not None

    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = BestSolutionSelection()
        with pytest.raises(Exception):
            selector.execute(invalid_input)

    def test_should_return_single_solution_when_list_has_one_solution(self, float_solution_factory):
        solution = float_solution_factory([1.0, 2.0])
        selector = BestSolutionSelection()
        
        result = selector.execute([solution])
        
        assert result == solution

    def test_should_select_non_dominated_solution(self, float_solution_factory):
        solution1 = float_solution_factory([1.0, 1.0])  # Dominates solution2
        solution2 = float_solution_factory([2.0, 2.0])
        
        selector = BestSolutionSelection()
        selection = selector.execute([solution1, solution2])
        
        assert selection == solution1

    def test_should_select_from_multiple_solutions(self, float_solution_factory):
        solutions = [
            float_solution_factory([1.0, 1.0]),
            float_solution_factory([0.0, 2.0]),
            float_solution_factory([0.5, 1.5]),  # Dominated
            float_solution_factory([0.0, 0.0]),
            float_solution_factory([1.0, 0.0])
        ]
        
        selector = BestSolutionSelection()
        selection = selector.execute(solutions)
        
        # Should return one of the non-dominated solutions
        assert selection in [solutions[0], solutions[1], solutions[3], solutions[4]]

class TestRandomSelection:
    """Tests for RandomSelection operator."""
    
    def test_should_initialize_correctly(self):
        assert RandomSelection() is not None

    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = RandomSelection()
        with pytest.raises(Exception):
            selector.execute(invalid_input)

    def test_should_return_single_solution_when_list_has_one_solution(self, float_solution_factory):
        solution = float_solution_factory([1.0, 2.0])
        selector = RandomSelection()
        
        result = selector.execute([solution])
        
        assert result == solution

    @patch('random.choice')
    def test_should_select_random_solution(self, mock_choice, float_solution_factory):
        solution1 = float_solution_factory([1.0, 1.0])
        solution2 = float_solution_factory([0.0, 2.0])
        
        # Test first solution being selected
        mock_choice.return_value = solution1
        selector = RandomSelection()
        selection = selector.execute([solution1, solution2])
        assert selection == solution1

    def test_should_return_solution_from_list(self, float_solution_factory):
        solutions = [
            float_solution_factory([1.0, 1.0]),
            float_solution_factory([0.0, 2.0]),
            float_solution_factory([0.5, 1.5])
        ]
        
        selector = RandomSelection()
        selection = selector.execute(solutions)
        
        assert selection in solutions

class TestDifferentialEvolutionSelection:
    """Tests for DifferentialEvolutionSelection operator."""

    def test_should_initialize_correctly(self):
        # Test default initialization
        selector = DifferentialEvolutionSelection()
        assert selector is not None
        assert selector.index_to_exclude is None
        
        # Test with index_to_exclude
        selector = DifferentialEvolutionSelection(index_to_exclude=5)
        assert selector.index_to_exclude == 5

    @pytest.mark.parametrize("invalid_input,expected_msg", [
        (None, "The front is null"),
        ([], "The front is empty"),
        ([1], "Differential evolution selection requires at least 4 solutions, got 1"),
        ([1, 2, 3], "Differential evolution selection requires at least 4 solutions, got 3")
    ])
    def test_should_raise_exception_for_invalid_input(self, invalid_input, expected_msg):
        selector = DifferentialEvolutionSelection()
        with pytest.raises(ValueError) as exc_info:
            selector.execute(invalid_input)
        assert expected_msg in str(exc_info.value)

    def test_should_raise_exception_if_not_enough_solutions(self, float_solution_factory):
        # Test with too few solutions
        solutions = [float_solution_factory([1.0, 1.0]) for _ in range(3)]  # Need at least 4 solutions
        selector = DifferentialEvolutionSelection()
        
        with pytest.raises(ValueError) as exc_info:
            selector.execute(solutions)
        assert "requires at least 4 solutions" in str(exc_info.value)
        
        # Test with index_to_exclude that leaves too few candidates
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(4)]
        selector = DifferentialEvolutionSelection(index_to_exclude=0)
        
        # The implementation doesn't actually raise an exception in this case
        # It will just select from the remaining 3 solutions
        result = selector.execute(solutions)
        assert len(result) == 3
        assert all(s in solutions[1:] for s in result)  # First solution should be excluded

    def test_should_return_three_different_solutions(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(10)]
        
        # Test without excluding any index
        selector = DifferentialEvolutionSelection()
        result = selector.execute(solutions)
        
        assert len(result) == 3
        assert all(s in solutions for s in result)
        # Instead of using a set, compare the objects directly
        assert result[0] is not result[1] and result[0] is not result[2] and result[1] is not result[2]
        
        # Test with index_to_exclude
        selector = DifferentialEvolutionSelection(index_to_exclude=0)
        result = selector.execute(solutions)
        
        assert len(result) == 3
        assert all(s in solutions[1:] for s in result)  # First solution should be excluded
        # Check all solutions are different
        assert result[0] is not result[1] and result[0] is not result[2] and result[1] is not result[2]

    @patch('random.sample')
    def test_should_select_random_solutions(self, mock_sample, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(10)]
        mock_sample.return_value = solutions[1:4]  # Return solutions 1,2,3
        
        # Test without excluding any index
        selector = DifferentialEvolutionSelection()
        result = selector.execute(solutions)
        
        assert len(result) == 3
        mock_sample.assert_called_once_with(solutions, 3)
        
        # Test with index_to_exclude
        mock_sample.reset_mock()
        mock_sample.return_value = solutions[1:4]  # Still return 1,2,3
        
        selector = DifferentialEvolutionSelection(index_to_exclude=0)
        result = selector.execute(solutions)
        
        assert len(result) == 3
        mock_sample.assert_called_once_with(solutions[1:], 3)  # Should exclude first solution

class TestNaryRandomSolutionSelection:
    """Tests for NaryRandomSolutionSelection operator."""

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_should_initialize_with_different_values_of_n(self, n):
        selector = NaryRandomSolutionSelection(n)
        assert selector is not None
        assert selector.number_of_solutions_to_be_returned == n

    def test_should_raise_exception_if_n_is_less_than_one(self):
        with pytest.raises(ValueError):
            NaryRandomSolutionSelection(0)

    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = NaryRandomSolutionSelection(1)
        with pytest.raises(Exception):
            selector.execute(invalid_input)

    def test_should_raise_exception_if_not_enough_solutions(self, float_solution_factory):
        solutions = [float_solution_factory([1.0, 1.0])]
        selector = NaryRandomSolutionSelection(2)
        
        with pytest.raises(Exception):
            selector.execute(solutions)

    @patch('random.sample')
    def test_should_return_correct_number_of_solutions(self, mock_sample, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(10)]
        n = 3
        mock_sample.return_value = solutions[:n]
        
        selector = NaryRandomSolutionSelection(n)
        result = selector.execute(solutions)
        
        assert len(result) == n
        mock_sample.assert_called_once_with(solutions, n)

    def test_should_return_all_solutions_if_n_equals_population_size(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(5)]
        
        selector = NaryRandomSolutionSelection(len(solutions))
        result = selector.execute(solutions)
        
        assert len(result) == len(solutions)
        assert all(s in solutions for s in result)

class TestRankingAndCrowdingDistanceSelection:
    """Tests for RankingAndCrowdingDistanceSelection operator."""

    def test_should_initialize_correctly(self):
        selector = RankingAndCrowdingDistanceSelection(5)
        assert selector is not None
        assert selector.max_population_size == 5  # Updated attribute name

    @pytest.mark.parametrize("invalid_input", [None, []])
    def test_should_raise_exception_for_invalid_input(self, invalid_input):
        selector = RankingAndCrowdingDistanceSelection(1)
        with pytest.raises(Exception):
            selector.execute(invalid_input)

    def test_should_raise_exception_if_max_population_size_is_invalid(self, float_solution_factory):
        solutions = [float_solution_factory([1.0, 1.0])]
        
        with pytest.raises(ValueError):
            selector = RankingAndCrowdingDistanceSelection(0)
            selector.execute(solutions)

    def test_should_return_all_solutions_if_fewer_than_max(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(3)]
        selector = RankingAndCrowdingDistanceSelection(5)
        
        result = selector.execute(solutions)
        
        assert len(result) == len(solutions)
        assert all(s in result for s in solutions)

    def test_should_return_correct_number_of_solutions(self, float_solution_factory):
        solutions = [float_solution_factory([float(i), float(i)]) for i in range(10)]
        max_population_size = 5
        selector = RankingAndCrowdingDistanceSelection(max_population_size)
        
        result = selector.execute(solutions)
        
        assert len(result) == max_population_size

    def test_should_work_with_single_objective(self, float_solution_factory):
        solutions = [float_solution_factory([float(i)]) for i in range(10)]
        selector = RankingAndCrowdingDistanceSelection(5)
        
        result = selector.execute(solutions)
        
        assert len(result) == 5

    def test_should_preserve_diversity(self, mixed_front_solutions):
        """Test that the selection maintains diversity by selecting from the non-dominated front."""
        # We'll select 3 solutions, which should all come from the non-dominated front
        selector = RankingAndCrowdingDistanceSelection(3)
        result = selector.execute(mixed_front_solutions)
        
        assert len(result) == 3
        
        # The first 3 solutions in mixed_front_solutions are non-dominated
        non_dominated = mixed_front_solutions[:3]
        assert all(s in non_dominated for s in result), \
            "All selected solutions should be from the non-dominated front"
        
        # The selection should include at least one extreme point
        extreme_points = [mixed_front_solutions[0], mixed_front_solutions[1]]
        assert any(s in extreme_points for s in result), \
            "At least one extreme point should be selected"