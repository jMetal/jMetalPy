"""Full-length runs checked against a known-good hypervolume floor, mirroring
tests/component/algorithm/multiobjective/test_nsgaii_integration.py but for the
component-based MOEA/D (classic and MOEA/D-DE).

Thresholds were calibrated by running each configuration across ~10 seeds and setting
the floor comfortably below the lowest observed value, not at the edge -- same
methodology as the NSGA-II/SMS-EMOA integration tests. rng= makes every run here
reproducible from the single fixed seed used, with no random.seed()/np.random.seed()
involved (see MODERNIZATION.md).
"""

import numpy as np
import pytest

from jmetal.component.algorithm.multiobjective.moead import build_moead, build_moead_de
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2, ZDT1

_SEED = 1


@pytest.mark.integration
class TestBuildMOEADReachesExpectedHypervolume:
    def test_should_reach_the_expected_hypervolume_on_zdt1(self):
        problem = ZDT1()
        algorithm = build_moead(
            problem,
            population_size=100,
            crossover=SBXCrossover(
                probability=1.0, distribution_index=20, rng=np.random.default_rng(_SEED)
            ),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(_SEED),
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
            rng=np.random.default_rng(_SEED),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.60

    def test_should_reach_the_expected_hypervolume_on_dtlz2(self):
        # population_size=91 matches a weight-vector file already bundled in
        # resources/MOEAD_weights/ (W3D_91.dat) -- no new resource needed.
        problem = DTLZ2()
        algorithm = build_moead(
            problem,
            population_size=91,
            crossover=SBXCrossover(
                probability=1.0, distribution_index=20, rng=np.random.default_rng(_SEED)
            ),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(_SEED),
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
            rng=np.random.default_rng(_SEED),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.38


@pytest.mark.integration
class TestBuildMOEADDEReachesExpectedHypervolume:
    def test_should_reach_the_expected_hypervolume_on_zdt1(self):
        problem = ZDT1()
        algorithm = build_moead_de(
            problem,
            population_size=100,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(_SEED),
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
            rng=np.random.default_rng(_SEED),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.50

    def test_should_reach_the_expected_hypervolume_on_dtlz2(self):
        problem = DTLZ2()
        algorithm = build_moead_de(
            problem,
            population_size=91,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(_SEED),
            ),
            termination=TerminationByEvaluations(max_evaluations=20000),
            rng=np.random.default_rng(_SEED),
        )

        algorithm.run()
        front = algorithm.result()

        hv = HyperVolume(reference_point=[1, 1, 1])
        value = hv.compute([solution.objectives for solution in front])

        assert value > 0.34
