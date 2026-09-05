"""Tests for jmetal.util.solution, in particular get_non_dominated_solutions().

get_non_dominated_solutions() had no dedicated tests before it was rewritten to
delegate to moocore.is_nondominated() instead of building a
NonDominatedSolutionsArchive one solution at a time -- these establish the
contract the old implementation had (duplicate handling, order, empty input) and
verify the new one still honors it.
"""

from jmetal.core.solution import FloatSolution
from jmetal.util.solution import get_non_dominated_solutions


def _solution(*objectives: float) -> FloatSolution:
    solution = FloatSolution([0.0] * len(objectives), [1.0] * len(objectives), len(objectives))
    solution.objectives = list(objectives)
    return solution


class TestGetNonDominatedSolutions:
    def test_empty_list_returns_empty_list(self):
        assert get_non_dominated_solutions([]) == []

    def test_single_solution_is_always_non_dominated(self):
        solution = _solution(1.0, 2.0)

        result = get_non_dominated_solutions([solution])

        assert result == [solution]

    def test_keeps_only_non_dominated_solutions(self):
        # (1, 5) and (5, 1) are mutually non-dominated; (3, 3) is dominated by
        # neither on its own, but (2, 2) dominates it outright.
        a = _solution(1.0, 5.0)
        b = _solution(5.0, 1.0)
        dominated = _solution(3.0, 3.0)
        dominator = _solution(2.0, 2.0)

        result = get_non_dominated_solutions([a, b, dominated, dominator])

        assert dominated not in result
        assert len(result) == 3
        assert a in result
        assert b in result
        assert dominator in result

    def test_a_solution_dominated_by_every_other_is_removed(self):
        best = _solution(0.0, 0.0)
        worst = _solution(1.0, 1.0)

        result = get_non_dominated_solutions([worst, best])

        assert result == [best]

    def test_all_mutually_non_dominated_solutions_are_kept(self):
        front = [_solution(0.0, 1.0), _solution(0.5, 0.5), _solution(1.0, 0.0)]

        result = get_non_dominated_solutions(front)

        assert len(result) == 3
        assert all(solution in result for solution in front)

    def test_duplicate_objectives_keep_only_the_first_occurrence(self):
        first = _solution(1.0, 1.0)
        duplicate = _solution(1.0, 1.0)

        result = get_non_dominated_solutions([first, duplicate])

        assert result == [first]

    def test_preserves_the_original_relative_order(self):
        # Non-dominated solutions should come back in the same relative order
        # they were given in, not sorted or reversed.
        c = _solution(0.0, 1.0)
        a = _solution(1.0, 0.0)
        b = _solution(0.5, 0.5)

        result = get_non_dominated_solutions([c, a, b])

        assert result == [c, a, b]

    def test_works_with_more_than_two_objectives(self):
        best = _solution(0.0, 0.0, 0.0)
        dominated = _solution(1.0, 1.0, 1.0)

        result = get_non_dominated_solutions([dominated, best])

        assert result == [best]

    def test_does_not_mutate_the_input_list(self):
        solutions = [_solution(1.0, 1.0), _solution(0.0, 0.0)]
        original = list(solutions)

        get_non_dominated_solutions(solutions)

        assert solutions == original
