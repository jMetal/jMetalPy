"""Tests for build_smsemoa()."""

import random

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.algorithm.multiobjective.smsemoa import build_smsemoa
from jmetal.component.catalogue.common.evaluation import SequentialEvaluationWithArchive
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive, NonDominatedSolutionsArchive


def _build(**overrides) -> EvolutionaryAlgorithm:
    problem = ZDT1()
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    return build_smsemoa(problem, 20, crossover, mutation, **overrides)


class TestBuildSMSEMOA:
    def test_returns_an_evolutionary_algorithm_named_smsemoa(self):
        algorithm = _build()

        assert isinstance(algorithm, EvolutionaryAlgorithm)
        assert algorithm.get_name() == "SMSEMOA"

    def test_default_termination_is_25000_evaluations(self):
        algorithm = _build()

        assert isinstance(algorithm.termination, TerminationByEvaluations)
        assert algorithm.termination.max_evaluations == 25000

    def test_default_variation_produces_a_single_offspring_per_generation(self):
        algorithm = _build()

        assert algorithm.variation.offspring_population_size() == 1

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

    def test_a_run_is_fully_reproducible_from_a_single_seed(self):
        # crossover/mutation are constructed outside build_smsemoa(), so they need
        # their own explicit rng too -- same requirement as build_nsgaii()/build_moead_de().
        # Unlike build_moead()/build_moead_de(), initial population creation goes
        # through the global random module (solutions_creation is deliberately
        # excluded from rng auto-threading -- see EvolutionaryAlgorithm's docstring),
        # so random.seed() is also needed right before run().
        def run_once():
            problem = ZDT1()
            crossover = SBXCrossover(
                probability=1.0, distribution_index=20, rng=np.random.default_rng(7)
            )
            mutation = PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(7),
            )
            algorithm = build_smsemoa(
                problem,
                20,
                crossover,
                mutation,
                termination=TerminationByEvaluations(max_evaluations=200),
                rng=np.random.default_rng(7),
            )
            random.seed(7)
            algorithm.run()
            return sorted(tuple(s.variables) for s in algorithm.result())

        assert run_once() == run_once()


class TestBuildSMSEMOAWithAnExternalArchive:
    def test_without_archive_evaluation_is_plain_sequential_evaluation(self):
        algorithm = _build()

        assert not isinstance(algorithm.evaluation, SequentialEvaluationWithArchive)

    def test_with_an_unbounded_archive_evaluation_feeds_it(self):
        archive = NonDominatedSolutionsArchive()

        algorithm = _build(archive=archive)

        assert isinstance(algorithm.evaluation, SequentialEvaluationWithArchive)
        assert algorithm.evaluation.archive is archive

    def test_with_a_bounded_crowding_archive_result_is_bounded_by_its_size(self):
        archive = CrowdingDistanceArchive(10)

        algorithm = _build(
            archive=archive, termination=TerminationByEvaluations(max_evaluations=100)
        )
        algorithm.run()

        assert len(algorithm.result()) <= 10
