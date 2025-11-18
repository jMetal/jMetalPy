import os
import sys
import numpy as np

# Ensure local src is preferred over an installed jmetal package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from jmetal.util.archive import distance_based_subset_selection_robust


class SimpleSolution:
    def __init__(self, objectives):
        self.objectives = list(objectives)
        self.attributes = {}

    def __deepcopy__(self, memo):
        # prevent deepcopy usage in code under test
        raise RuntimeError("deepcopy not allowed")


def _make_solutions(mat):
    return [SimpleSolution(row) for row in mat]


def test_l2_vectorized_equivalence():
    rng = np.random.default_rng(42)
    mat = rng.random((50, 3))
    sols = _make_solutions(mat)

    sel1 = distance_based_subset_selection_robust(sols, 10, random_seed=123, use_vectorized=True)
    sel2 = distance_based_subset_selection_robust(sols, 10, random_seed=123, use_vectorized=False)

    set1 = {tuple(s.objectives) for s in sel1}
    set2 = {tuple(s.objectives) for s in sel2}

    assert set1 == set2


def test_reproducibility_with_seed():
    mat = np.random.RandomState(0).rand(40, 4)
    sols = _make_solutions(mat)

    sel_a = distance_based_subset_selection_robust(sols, 8, random_seed=2025, use_vectorized=True)
    sel_b = distance_based_subset_selection_robust(sols, 8, random_seed=2025, use_vectorized=True)

    assert [tuple(s.objectives) for s in sel_a] == [tuple(s.objectives) for s in sel_b]


def test_crowding_selection_no_deepcopy():
    # 2-objective case triggers crowding selection; ensure no deepcopy is invoked
    rng = np.random.default_rng(1)
    mat = rng.random((20, 2))
    sols = _make_solutions(mat)

    sel = distance_based_subset_selection_robust(sols, 5, random_seed=7, use_vectorized=True)
    assert len(sel) == 5
