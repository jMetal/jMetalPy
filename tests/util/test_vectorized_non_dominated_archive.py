import numpy as np

from jmetal.util.archive import NonDominatedSolutionsArchive, VectorizedNonDominatedSolutionsArchive
from jmetal.core.solution import BinarySolution


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

    na.add(s1); na.add(s2)
    va.add(s1); va.add(s2)

    # new solution dominates s1 and s2
    s_new = make_solution([0.5, 0.4])
    assert na.add(s_new)
    assert va.add(s_new)

    assert extract_objectives(na) == extract_objectives(va)
