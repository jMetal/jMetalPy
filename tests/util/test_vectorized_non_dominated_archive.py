import numpy as np
import pytest

from jmetal.core.solution import BinarySolution
from jmetal.util.archive import NonDominatedSolutionsArchive, VectorizedNonDominatedSolutionsArchive


def make_solution(objs):
    s = BinarySolution(number_of_variables=1, number_of_objectives=len(objs))
    s.objectives = list(objs)
    return s


def extract_objectives(archive):
    return {tuple(s.objectives) for s in archive.solution_list}


def test_vectorized_matches_original_simple():
    na = NonDominatedSolutionsArchive()
    va = VectorizedNonDominatedSolutionsArchive()

    sols = [make_solution([1.0, 2.0]), make_solution([2.0, 1.0]), make_solution([1.5, 1.5])]

    for s in sols:
        assert na.add(s)
        assert va.add(s)

    assert extract_objectives(na) == extract_objectives(va)


def test_vectorized_rejects_dominated_and_duplicates():
    na = NonDominatedSolutionsArchive()
    va = VectorizedNonDominatedSolutionsArchive()

    a = make_solution([1.0, 1.0])
    b = make_solution([2.0, 2.0])
    c = make_solution([1.0, 1.0])  # duplicate of a

    assert na.add(a)
    assert va.add(a)

    # b is dominated by a -> should be rejected
    assert na.add(b) is False
    assert va.add(b) is False

    # duplicate should be rejected
    assert na.add(c) is False
    assert va.add(c) is False

    assert extract_objectives(na) == extract_objectives(va)


def test_vectorized_removes_existing_when_new_dominates():
    na = NonDominatedSolutionsArchive()
    va = VectorizedNonDominatedSolutionsArchive()

    # existing solutions
    s1 = make_solution([2.0, 0.5])
    s2 = make_solution([1.5, 1.5])

    na.add(s1)
    na.add(s2)
    va.add(s1)
    va.add(s2)

    # new solution dominates s1 and s2
    s_new = make_solution([0.5, 0.4])
    assert na.add(s_new)
    assert va.add(s_new)

    assert extract_objectives(na) == extract_objectives(va)


@pytest.mark.parametrize("number_of_objectives", [2, 3, 5])
@pytest.mark.parametrize("use_batch", [False, True])
def test_should_both_archives_keep_the_same_solutions_on_random_data(
    number_of_objectives, use_batch
):
    rng = np.random.default_rng(7)
    points = np.round(rng.random((400, number_of_objectives)), 3)  # rounding creates duplicates
    solutions = [make_solution(row) for row in points]
    archives = [NonDominatedSolutionsArchive(), VectorizedNonDominatedSolutionsArchive()]

    for archive in archives:
        if use_batch:
            archive.add_batch(solutions)
        else:
            for solution in solutions:
                archive.add(solution)

    assert extract_objectives(archives[0]) == extract_objectives(archives[1])
    assert archives[0].size() == archives[1].size()


@pytest.mark.parametrize("number_of_objectives", [2, 3])
@pytest.mark.parametrize(
    "points",
    [
        [(0.0, 1.0), (5.0, 1.0 - 5e-11)],  # better by less than the tolerance in one objective
        [(5.0, 1.0), (0.0, 1.0 + 5e-11)],
    ],
)
def test_should_keep_solutions_that_are_better_by_less_than_the_tolerance(
    number_of_objectives, points
):
    padding = [0.0] * (number_of_objectives - 2)
    for archive in (NonDominatedSolutionsArchive(), VectorizedNonDominatedSolutionsArchive()):
        for point in points:
            archive.add(make_solution(list(point) + padding))

        assert archive.size() == 2


@pytest.mark.parametrize("number_of_objectives", [2, 3, 5])
def test_should_both_archives_agree_on_points_closer_than_the_tolerance(number_of_objectives):
    rng = np.random.default_rng(11)
    base = np.round(rng.random((150, number_of_objectives)), 2)
    noise = rng.uniform(-2e-10, 2e-10, base.shape)  # around objective_tolerance (1e-10)
    solutions = [make_solution(row) for row in np.vstack([base, base + noise])]
    archives = [NonDominatedSolutionsArchive(), VectorizedNonDominatedSolutionsArchive()]

    for archive in archives:
        for solution in solutions:
            archive.add(solution)

    assert extract_objectives(archives[0]) == extract_objectives(archives[1])


@pytest.mark.parametrize("number_of_objectives", [2, 3])
def test_should_vectorized_result_not_depend_on_insertion_order_for_a_dominated_chain(
    number_of_objectives,
):
    padding = [0.0] * (number_of_objectives - 2)
    chain = [[0.0, 0.0] + padding, [0.0, 8e-11] + padding, [0.0, 1.6e-10] + padding]
    results = set()

    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1]):
        archive = VectorizedNonDominatedSolutionsArchive()
        for index in order:
            archive.add(make_solution(chain[index]))
        results.add(frozenset(extract_objectives(archive)))

    assert results == {frozenset({tuple(chain[0])})}
