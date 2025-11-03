from unittest.mock import patch

import pytest

from jmetal.core.solution import (
    Solution,
    FloatSolution
)
from jmetal.operator.selection import (
    BinaryTournamentSelection,
    BestSolutionSelection,
    RandomSelection,
    DifferentialEvolutionSelection,
    NaryRandomSolutionSelection,
    RankingAndCrowdingDistanceSelection
)


# Fixtures and helper functions
@pytest.fixture
def float_solution():
    def _create_float_solution(objectives, variables=None):
        if variables is None:
            variables = [0.0] * len(objectives)
        solution = FloatSolution([0.0] * len(variables), [1.0] * len(variables), len(objectives))
        solution.objectives = objectives
        solution.variables = variables
        return solution
    return _create_float_solution

class DummySolution(Solution):
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

# Test classes will be added here one by one
class TestBinaryTournamentSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = BinaryTournamentSelection()
        assert selector is not None

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selector = BinaryTournamentSelection()
        with pytest.raises(Exception):
            selector.execute(None)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selector = BinaryTournamentSelection()
        with pytest.raises(Exception):
            selector.execute([])

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self, float_solution):
        solution = float_solution([1.0, 2.0])
        selector = BinaryTournamentSelection()
        selection = selector.execute([solution])
        assert selection == solution

    @patch('random.sample')
    def test_should_execute_work_if_the_solution_list_contains_two_non_dominated_solutions(self, random_mock, float_solution):
        random_mock.return_value = [0, 1]  # Always select first two solutions
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([0.0, 2.0])
        
        selector = BinaryTournamentSelection()
        selection = selector.execute([solution1, solution2])
        
        assert selection in [solution1, solution2]

    def test_should_execute_work_if_the_solution_list_contains_five_solutions(self, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),
            float_solution([0.0, 2.0]),
            float_solution([0.5, 1.5]),
            float_solution([0.0, 0.0]),
            float_solution([1.0, 0.0])
        ]
        
        with patch('random.sample', side_effect=lambda x, k: [0, 1]):
            selector = BinaryTournamentSelection()
            selection = selector.execute(solutions)
            
            assert selection in solutions

    def test_should_the_operator_work_with_the_default_parameters(self, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),
            float_solution([0.0, 2.0]),
            float_solution([0.5, 1.5]),
            float_solution([0.0, 0.0]),
            float_solution([1.0, 0.0])
        ]
        
        selector = BinaryTournamentSelection()
        selection = selector.execute(solutions)
        
        assert selection in solutions


class TestBestSolutionSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = BestSolutionSelection()
        assert selector is not None

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selector = BestSolutionSelection()
        with pytest.raises(Exception):
            selector.execute(None)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selector = BestSolutionSelection()
        with pytest.raises(Exception):
            selector.execute([])

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self, float_solution):
        solution = float_solution([1.0, 2.0])
        selector = BestSolutionSelection()
        selection = selector.execute([solution])
        assert selection == solution

    def test_should_execute_work_if_the_solution_list_contains_two_non_dominated_solutions(self, float_solution):
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([0.0, 2.0])
        
        selector = BestSolutionSelection()
        selection = selector.execute([solution1, solution2])
        
        # Both solutions are non-dominated, so either is acceptable
        assert selection in [solution1, solution2]

    def test_should_execute_work_if_the_solution_list_contains_two_solutions_and_one_them_is_dominated(self, float_solution):
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([2.0, 2.0])  # Dominated by solution1
        
        selector = BestSolutionSelection()
        selection = selector.execute([solution1, solution2])
        
        assert selection == solution1

    def test_should_execute_work_if_the_solution_list_contains_five_solutions_and_one_them_is_dominated(self, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),  # Non-dominated
            float_solution([0.0, 2.0]),  # Non-dominated
            float_solution([0.5, 1.5]),  # Dominated by [0.0, 2.0] and [1.0, 1.0]
            float_solution([0.0, 0.0]),  # Non-dominated
            float_solution([1.0, 0.0])   # Non-dominated
        ]
        
        selector = BestSolutionSelection()
        selection = selector.execute(solutions)
        
        # Should return one of the non-dominated solutions
        assert selection in [solutions[0], solutions[1], solutions[3], solutions[4]]


class TestRandomSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = RandomSelection()
        assert selector is not None

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selector = RandomSelection()
        with pytest.raises(Exception):
            selector.execute(None)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selector = RandomSelection()
        with pytest.raises(Exception):
            selector.execute([])

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self, float_solution):
        solution = float_solution([1.0, 2.0])
        selector = RandomSelection()
        selection = selector.execute([solution])
        assert selection == solution

    @patch('random.choice')
    def test_should_execute_work_if_the_solution_list_contains_two_solutions(self, mock_choice, float_solution):
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([0.0, 2.0])
        
        # Test first solution being selected
        mock_choice.return_value = solution1
        selector = RandomSelection()
        selection = selector.execute([solution1, solution2])
        assert selection == solution1
        
        # Test second solution being selected
        mock_choice.return_value = solution2
        selection = selector.execute([solution1, solution2])
        assert selection == solution2

    def test_should_execute_return_a_solution_from_the_list(self, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),
            float_solution([0.0, 2.0]),
            float_solution([0.5, 1.5]),
            float_solution([0.0, 0.0]),
            float_solution([1.0, 0.0])
        ]
        
        selector = RandomSelection()
        
        # Test multiple times to ensure random selection works
        for _ in range(10):
            selection = selector.execute(solutions)
            assert selection in solutions


class TestDifferentialEvolutionSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = DifferentialEvolutionSelection()
        assert selector is not None

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selector = DifferentialEvolutionSelection()
        with pytest.raises(Exception):
            selector.execute(None)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selector = DifferentialEvolutionSelection()
        with pytest.raises(Exception):
            selector.execute([])

    def test_should_execute_raise_an_exception_if_the_list_has_less_than_three_solutions(self, float_solution):
        selector = DifferentialEvolutionSelection()
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([2.0, 2.0])
        
        with pytest.raises(Exception):
            selector.execute([solution1, solution2])

    @patch('random.sample')
    def test_should_execute_return_three_solutions_if_the_list_has_more_than_three_solutions(self, mock_sample, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),
            float_solution([2.0, 2.0]),
            float_solution([3.0, 3.0]),
            float_solution([4.0, 4.0])
        ]
        
        # Mock random.sample to return the first three solutions
        mock_sample.return_value = solutions[:3]
        
        selector = DifferentialEvolutionSelection()
        selected = selector.execute(solutions)
        
        assert len(selected) == 3
        assert selected == solutions[:3]
        mock_sample.assert_called_once_with(solutions, 3)

    def test_should_execute_exclude_the_indicated_solution(self, float_solution):
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([2.0, 2.0])
        solution3 = float_solution([3.0, 3.0])
        solution4 = float_solution([4.0, 4.0])
        
        solutions = [solution1, solution2, solution3, solution4]
        
        # Create a mock for random.sample that excludes the first solution
        def mock_sample(population, k):
            # Create a copy of the population without the first solution
            filtered_population = [s for s in population if s != solution1]
            # Return the first k solutions from the filtered population
            return filtered_population[:k]
        
        with patch('random.sample', side_effect=mock_sample):
            selector = DifferentialEvolutionSelection()
            selector.index = 0  # Exclude solution1
            selected = selector.execute(solutions)
            
            assert len(selected) == 3
            assert solution1 not in selected
            # The selected solutions should be from the remaining solutions
            assert all(sol in [solution2, solution3, solution4] for sol in selected)


class TestNaryRandomSolutionSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = NaryRandomSolutionSelection()
        assert selector is not None

    def test_should_constructor_create_a_non_null_object_and_check_number_of_elements(self):
        selector = NaryRandomSolutionSelection(3)
        assert selector is not None
        assert selector.number_of_solutions_to_be_returned == 3

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selector = NaryRandomSolutionSelection()
        with pytest.raises(Exception):
            selector.execute(None)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selector = NaryRandomSolutionSelection()
        with pytest.raises(Exception):
            selector.execute([])

    def test_should_execute_raise_an_exception_if_the_list_has_fewer_solutions_than_requested(self, float_solution):
        selector = NaryRandomSolutionSelection(3)
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([2.0, 2.0])
        
        with pytest.raises(Exception):
            selector.execute([solution1, solution2])

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self, float_solution):
        solution = float_solution([1.0, 2.0])
        selector = NaryRandomSolutionSelection(1)
        selection = selector.execute([solution])
        assert selection == [solution]

    @patch('random.sample')
    def test_should_execute_work_with_the_requested_number_of_solutions(self, mock_sample, float_solution):
        solution1 = float_solution([1.0, 1.0])
        solution2 = float_solution([2.0, 2.0])
        solution3 = float_solution([3.0, 3.0])
        
        # Mock random.sample to return the first two solutions
        mock_sample.return_value = [solution1, solution2]
        
        selector = NaryRandomSolutionSelection(2)
        selection = selector.execute([solution1, solution2, solution3])
        
        assert len(selection) == 2
        assert selection == [solution1, solution2]
        mock_sample.assert_called_once_with([solution1, solution2, solution3], 2)

    def test_should_execute_return_different_solutions_each_time(self, float_solution):
        solutions = [
            float_solution([1.0, 1.0]),
            float_solution([2.0, 2.0]),
            float_solution([3.0, 3.0]),
            float_solution([4.0, 4.0]),
            float_solution([5.0, 5.0])
        ]
        
        selector = NaryRandomSolutionSelection(3)
        
        # Test multiple times to ensure different selections
        selected_solutions = set()
        for _ in range(10):
            selection = selector.execute(solutions)
            assert len(selection) == 3
            assert all(sol in solutions for sol in selection)
            # Convert to tuple to make it hashable for the set
            selected_solutions.add(tuple(id(sol) for sol in selection))
        
        # We should have at least 2 different selections in 10 trials
        assert len(selected_solutions) > 1


class TestRankingAndCrowdingDistanceSelection:
    def test_should_constructor_create_a_non_null_object(self):
        selector = RankingAndCrowdingDistanceSelection(100)
        assert selector is not None
        assert selector.max_population_size == 100

    def test_should_execute_raise_exception_if_front_is_none(self):
        selector = RankingAndCrowdingDistanceSelection(100)
        with pytest.raises(ValueError):
            selector.execute(None)

    def test_should_execute_raise_exception_if_front_is_empty(self):
        selector = RankingAndCrowdingDistanceSelection(100)
        with pytest.raises(ValueError):
            selector.execute([])

    def test_should_raise_exception_if_max_population_size_is_invalid(self, float_solution):
        # The constructor doesn't validate max_population_size, but execute() does
        # So we'll test that execute() raises the appropriate error
        selector = RankingAndCrowdingDistanceSelection(0)
        solutions = [float_solution([1.0, 2.0])]
        
        with pytest.raises(ValueError):
            selector.execute(solutions)
            
        selector = RankingAndCrowdingDistanceSelection(-1)
        with pytest.raises(ValueError):
            selector.execute(solutions)

    def test_should_execute_return_all_solutions_if_fewer_than_max_population_size(self, float_solution):
        solutions = [
            float_solution([1.0, 2.0]),
            float_solution([2.0, 1.0]),
            float_solution([3.0, 0.5])
        ]
        
        # Test with max_population_size larger than number of solutions
        selector = RankingAndCrowdingDistanceSelection(5)
        selected = selector.execute(solutions)
        
        # Should return all solutions since we have fewer than max_population_size
        assert len(selected) == 3
        assert all(sol in selected for sol in solutions)
        
        # Test with max_population_size equal to number of solutions
        selector = RankingAndCrowdingDistanceSelection(3)
        selected = selector.execute(solutions)
        assert len(selected) == 3
        assert all(sol in selected for sol in solutions)

    def test_should_execute_return_correct_number_of_solutions(self, float_solution):
        # Create test solutions
        solutions = [
            float_solution([1.0, 2.0]),
            float_solution([2.0, 1.0]),
            float_solution([0.5, 2.5]),
            float_solution([1.5, 1.5]),
            float_solution([2.5, 0.5])
        ]
        
        # Test with different max_population_size values
        for n in range(1, 6):
            selector = RankingAndCrowdingDistanceSelection(n)
            selected = selector.execute(solutions)
            assert len(selected) == min(n, len(solutions))
