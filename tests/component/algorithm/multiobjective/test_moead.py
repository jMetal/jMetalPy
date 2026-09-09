"""Tests for build_moead() and build_moead_de()."""

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.algorithm.multiobjective.moead import build_moead, build_moead_de
from jmetal.component.catalogue.common.evaluation import SequentialEvaluationWithArchive
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.aggregation_function import PenaltyBoundaryIntersection, Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive, NonDominatedSolutionsArchive

_POPULATION_SIZE = 20


def _build_classic(**overrides) -> EvolutionaryAlgorithm:
    problem = ZDT1()
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    return build_moead(problem, _POPULATION_SIZE, crossover, mutation, **overrides)


def _build_de(**overrides) -> EvolutionaryAlgorithm:
    problem = ZDT1()
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    return build_moead_de(problem, _POPULATION_SIZE, mutation, **overrides)


class TestBuildMOEAD:
    def test_returns_an_evolutionary_algorithm_named_moead(self):
        algorithm = _build_classic()

        assert isinstance(algorithm, EvolutionaryAlgorithm)
        assert algorithm.get_name() == "MOEAD"

    def test_default_termination_is_25000_evaluations(self):
        algorithm = _build_classic()

        assert isinstance(algorithm.termination, TerminationByEvaluations)
        assert algorithm.termination.max_evaluations == 25000

    def test_default_variation_produces_a_single_offspring_per_generation(self):
        algorithm = _build_classic()

        assert algorithm.variation.offspring_population_size() == 1

    def test_default_aggregation_function_is_pbi(self):
        algorithm = _build_classic()

        assert isinstance(algorithm.replacement.aggregation_function, PenaltyBoundaryIntersection)

    def test_a_short_run_terminates_and_returns_the_configured_population_size(self):
        algorithm = _build_classic(termination=TerminationByEvaluations(max_evaluations=100))

        algorithm.run()

        assert len(algorithm.result()) == _POPULATION_SIZE

    def test_overriding_termination_is_honored(self):
        custom_termination = TerminationByEvaluations(max_evaluations=40)

        algorithm = _build_classic(termination=custom_termination)

        assert algorithm.termination is custom_termination

    def test_overriding_aggregation_function_is_honored(self):
        custom_aggregation = Tschebycheff(dimension=2)

        algorithm = _build_classic(aggregation_function=custom_aggregation)

        assert algorithm.replacement.aggregation_function is custom_aggregation

    def test_rng_is_shared_with_the_template(self):
        rng = np.random.default_rng(123)

        algorithm = _build_classic(rng=rng)

        assert algorithm.rng is rng

    def test_rng_is_shared_with_the_moead_context(self):
        rng = np.random.default_rng(123)

        algorithm = _build_classic(rng=rng)

        assert algorithm.selection.context.rng is rng
        assert algorithm.replacement.context.rng is rng

    def test_a_run_is_fully_reproducible_from_a_single_seed(self):
        # Unlike build_nsgaii()/build_smsemoa(), no random.seed()/np.random.seed()
        # should be necessary -- every MOEA/D-specific random draw goes through rng.
        def run_once():
            problem = ZDT1()
            algorithm = build_moead(
                problem,
                _POPULATION_SIZE,
                crossover=SBXCrossover(
                    probability=1.0, distribution_index=20, rng=np.random.default_rng(7)
                ),
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables(),
                    distribution_index=20,
                    rng=np.random.default_rng(7),
                ),
                termination=TerminationByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(7),
            )
            algorithm.run()
            return sorted(tuple(s.variables) for s in algorithm.result())

        assert run_once() == run_once()


class TestBuildMOEADWithAnExternalArchive:
    def test_without_archive_evaluation_is_plain_sequential_evaluation(self):
        algorithm = _build_classic()

        assert not isinstance(algorithm.evaluation, SequentialEvaluationWithArchive)

    def test_with_an_unbounded_archive_evaluation_feeds_it(self):
        archive = NonDominatedSolutionsArchive()

        algorithm = _build_classic(archive=archive)

        assert isinstance(algorithm.evaluation, SequentialEvaluationWithArchive)
        assert algorithm.evaluation.archive is archive

    def test_with_a_bounded_crowding_archive_result_is_bounded_by_its_size(self):
        archive = CrowdingDistanceArchive(10)

        algorithm = _build_classic(
            archive=archive, termination=TerminationByEvaluations(max_evaluations=100)
        )
        algorithm.run()

        assert len(algorithm.result()) <= 10


class TestBuildMOEADDE:
    def test_returns_an_evolutionary_algorithm_named_moead_de(self):
        algorithm = _build_de()

        assert isinstance(algorithm, EvolutionaryAlgorithm)
        assert algorithm.get_name() == "MOEAD-DE"

    def test_default_aggregation_function_is_tschebycheff(self):
        algorithm = _build_de()

        assert isinstance(algorithm.replacement.aggregation_function, Tschebycheff)

    def test_default_variation_uses_differential_evolution_crossover(self):
        from jmetal.operator.crossover import DifferentialEvolutionCrossover

        algorithm = _build_de()

        assert isinstance(algorithm.variation.crossover, DifferentialEvolutionCrossover)

    def test_cr_and_f_are_forwarded_to_the_crossover_operator(self):
        algorithm = _build_de(cr=0.3, f=0.7)

        assert algorithm.variation.crossover.CR == 0.3
        assert algorithm.variation.crossover.F == 0.7

    def test_a_short_run_terminates_and_returns_the_configured_population_size(self):
        algorithm = _build_de(termination=TerminationByEvaluations(max_evaluations=100))

        algorithm.run()

        assert len(algorithm.result()) == _POPULATION_SIZE

    def test_a_run_is_fully_reproducible_from_a_single_seed(self):
        # mutation is constructed outside build_moead_de(), so it needs its own
        # explicit rng= too -- same requirement as build_nsgaii()/build_smsemoa(),
        # unaffected by MOEA/D's own single-seed-reproducibility guarantee for the
        # pieces build_moead_de() constructs itself (DE crossover, MOEADContext).
        def run_once():
            problem = ZDT1()
            mutation = PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
                rng=np.random.default_rng(7),
            )
            algorithm = build_moead_de(
                problem,
                _POPULATION_SIZE,
                mutation,
                termination=TerminationByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(7),
            )
            algorithm.run()
            return sorted(tuple(s.variables) for s in algorithm.result())

        assert run_once() == run_once()
