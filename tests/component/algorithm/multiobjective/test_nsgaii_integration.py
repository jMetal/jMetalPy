"""Full-length runs checked against a known-good hypervolume floor, mirroring
tests/algorithm/test_algorithm_integration.py but for the component-based NSGA-II.
"""

import pytest

from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2, ZDT1, ZDT4
from jmetal.util.archive import CrowdingDistanceArchive, NonDominatedSolutionsArchive


@pytest.mark.integration
class TestBuildNSGAIIReachesExpectedHypervolume:
    def test_should_reach_the_expected_hypervolume_on_zdt4_with_a_crowding_archive(self):
        # ZDT4 is multi-modal (many local Pareto fronts); a bounded external
        # crowding-distance archive keeps every evaluated solution as a candidate
        # regardless of what the population/replacement strategy discards, guarding
        # against convergence on a local front.
        problem = ZDT4()
        archive = CrowdingDistanceArchive(maximum_size=100)
        algorithm = build_nsgaii(
            problem,
            population_size=100,
            offspring_population_size=100,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
            archive=archive,
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.60

    def test_should_reach_the_expected_hypervolume_on_zdt1_with_offspring_size_one(self):
        # A steady-state configuration: one offspring per generation instead of a
        # full population's worth, exercising CrossoverAndMutationVariation's
        # mating-pool-size arithmetic at its smallest useful offspring size.
        problem = ZDT1()
        algorithm = build_nsgaii(
            problem,
            population_size=100,
            offspring_population_size=1,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.63

    def test_should_reach_the_expected_hypervolume_on_dtlz2_with_an_unbounded_archive(self):
        # An unbounded archive keeps every non-dominated solution ever evaluated,
        # growing into the thousands over a full run (Archive.add_batch(), moocore
        # under the hood, keeps updating it fast: ~1.4s here rather than the ~43s a
        # one-insertion-at-a-time update took before it existed). result() then
        # reduces it to the population size via distance-based subset selection,
        # so the front stays representable and doesn't just grow unbounded too.
        problem = DTLZ2()
        archive = NonDominatedSolutionsArchive()
        algorithm = build_nsgaii(
            problem,
            population_size=100,
            offspring_population_size=100,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(), distribution_index=20
            ),
            termination=TerminationByEvaluations(max_evaluations=40000),
            archive=archive,
        )

        algorithm.run()
        front = algorithm.result()

        assert len(archive.solution_list) > 100
        assert len(front) == 100

        hv = HyperVolume(reference_point=[1, 1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.35
