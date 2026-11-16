"""Regression tests: the vectorized/moocore fast paths must not be taken for
DominanceComparator subclasses such as GDominanceComparator, whose dominance
relation they do not reproduce."""

import pytest

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import (
    NonDominatedSolutionsArchive,
    VectorizedNonDominatedSolutionsArchive,
)
from jmetal.util.comparator import GDominanceComparator
from jmetal.util.ranking import FastNonDominatedRanking

REFERENCE_POINT = [0.5, 0.5]
ARCHIVE_CLASSES = [NonDominatedSolutionsArchive, VectorizedNonDominatedSolutionsArchive]


def _solution(*objectives: float) -> FloatSolution:
    solution = FloatSolution([0.0], [1.0], len(objectives))
    solution.objectives = list(objectives)
    return solution


def _in_region_and_outside() -> list[FloatSolution]:
    # (0.4, 0.4) lies in the region of interest, (0.1, 0.9) does not: under
    # g-dominance the first dominates the second, under Pareto they are incomparable.
    return [_solution(0.4, 0.4), _solution(0.1, 0.9)]


class TestArchiveWithGDominanceComparator:
    @pytest.mark.parametrize("archive_class", ARCHIVE_CLASSES)
    def test_should_add_honour_g_dominance(self, archive_class):
        archive = archive_class(GDominanceComparator(REFERENCE_POINT))

        for solution in _in_region_and_outside():
            archive.add(solution)

        assert [s.objectives for s in archive.solution_list] == [[0.4, 0.4]]

    @pytest.mark.parametrize("archive_class", ARCHIVE_CLASSES)
    def test_should_add_batch_honour_g_dominance(self, archive_class):
        archive = archive_class(GDominanceComparator(REFERENCE_POINT))

        archive.add_batch(_in_region_and_outside())

        assert [s.objectives for s in archive.solution_list] == [[0.4, 0.4]]


class TestFastNonDominatedRankingWithGDominanceComparator:
    def test_should_compute_ranking_honour_g_dominance(self):
        ranking = FastNonDominatedRanking(GDominanceComparator(REFERENCE_POINT))

        ranking.compute_ranking(_in_region_and_outside())

        fronts = [[s.objectives for s in front] for front in ranking.ranked_sublists]
        assert fronts == [[[0.4, 0.4]], [[0.1, 0.9]]]
