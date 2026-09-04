"""Tests for build_nsgaii()."""

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1


def _build(**overrides) -> EvolutionaryAlgorithm:
    problem = ZDT1()
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    return build_nsgaii(problem, 20, 20, crossover, mutation, **overrides)


class TestBuildNSGAII:
    def test_returns_an_evolutionary_algorithm_named_nsgaii(self):
        algorithm = _build()

        assert isinstance(algorithm, EvolutionaryAlgorithm)
        assert algorithm.get_name() == "NSGAII"

    def test_default_termination_is_25000_evaluations(self):
        algorithm = _build()

        assert isinstance(algorithm.termination, TerminationByEvaluations)
        assert algorithm.termination.max_evaluations == 25000

    def test_default_selection_mating_pool_size_matches_variation(self):
        algorithm = _build()

        assert algorithm.selection.mating_pool_size == algorithm.variation.mating_pool_size()

    def test_a_short_run_terminates_and_returns_the_configured_population_size(self):
        algorithm = _build(termination=TerminationByEvaluations(max_evaluations=100))

        algorithm.run()

        assert len(algorithm.result()) == 20

    def test_overriding_termination_is_honored(self):
        custom_termination = TerminationByEvaluations(max_evaluations=40)

        algorithm = _build(termination=custom_termination)

        assert algorithm.termination is custom_termination

    def test_rng_is_shared_with_the_template(self):
        rng = np.random.default_rng(123)

        algorithm = _build(rng=rng)

        assert algorithm.rng is rng

    def test_handles_an_odd_offspring_population_size(self):
        # mating_pool_size = parents * ceil(offspring_size / children); for SBX
        # (2 parents, 2 children) and an odd offspring size that rounds up, so the
        # mating pool stays evenly divisible by the number of parents.
        algorithm = build_nsgaii(
            ZDT1(),
            population_size=5,
            offspring_population_size=5,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            mutation=PolynomialMutation(probability=0.1, distribution_index=20),
            termination=TerminationByEvaluations(max_evaluations=6),
        )

        algorithm.run()

        assert len(algorithm.result()) == 5
