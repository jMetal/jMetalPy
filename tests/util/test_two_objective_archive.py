import numpy as np

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import NonDominatedSolutionsArchive, VectorizedNonDominatedSolutionsArchive


def make_solution(obj0, obj1):
    s = FloatSolution(lower_bound=[], upper_bound=[], number_of_objectives=2)
    s.objectives = [float(obj0), float(obj1)]
    return s


def test_two_objective_parity_with_original():
    rng = np.random.default_rng(1234)
    pts = rng.random((300, 2))

    vec = VectorizedNonDominatedSolutionsArchive()
    orig = NonDominatedSolutionsArchive()

    for x, y in pts:
        s = make_solution(x, y)
        vec.add(s)
        # use a fresh copy for original to avoid shared object references
        t = make_solution(x, y)
        orig.add(t)

    v_objs = np.asarray([s.objectives for s in vec.solution_list])
    o_objs = np.asarray([s.objectives for s in orig.solution_list])

    # compare sets ignoring order
    v_sorted = np.sort(v_objs, axis=0)
    o_sorted = np.sort(o_objs, axis=0)

    assert v_sorted.shape == o_sorted.shape
    assert np.allclose(
        np.sort(v_objs.view(float).reshape(v_objs.shape), axis=0),
        np.sort(o_objs.view(float).reshape(o_objs.shape), axis=0),
    )


def test_two_objective_invariant_ordering():
    rng = np.random.default_rng(42)
    pts = rng.random((200, 2))

    vec = VectorizedNonDominatedSolutionsArchive()
    for x, y in pts:
        vec.add(make_solution(x, y))

    objs = np.asarray([s.objectives for s in vec.solution_list], dtype=float)
    if objs.size == 0:
        return

    tol = vec.objective_tolerance
    # obj0 ascending
    assert np.all(objs[:-1, 0] <= objs[1:, 0] + tol)
    # obj1 non-increasing (to maintain Pareto front)
    assert np.all(objs[:-1, 1] >= objs[1:, 1] - tol)


def test_two_objective_duplicates_and_dominance():
    vec = VectorizedNonDominatedSolutionsArchive()

    a = make_solution(0.5, 0.5)
    assert vec.add(a) is True

    # duplicate
    b = make_solution(0.5, 0.5)
    assert vec.add(b) is False

    # dominating new solution (same obj0, better obj1)
    c = make_solution(0.5, 0.4)
    assert vec.add(c) is True
    # now archive should contain c but not a
    objs = [s.objectives for s in vec.solution_list]
    assert any(np.allclose(o, [0.5, 0.4]) for o in objs)
    assert not any(np.allclose(o, [0.5, 0.5]) for o in objs)

    # dominated candidate
    d = make_solution(0.6, 0.6)
    assert vec.add(d) is False


def test_should_preserve_results_when_switching_to_vectorized_insertion():
    points = [(index / 255, 1 - index / 255) for index in range(256)]
    points.extend([(0.25, 0.25), (0.9, 0.9)])
    archives = [NonDominatedSolutionsArchive(), VectorizedNonDominatedSolutionsArchive()]

    for archive in archives:
        for point in points:
            archive.add(make_solution(*point))

    expected = {tuple(solution.objectives) for solution in archives[0].solution_list}
    actual = {tuple(solution.objectives) for solution in archives[1].solution_list}

    assert actual == expected
