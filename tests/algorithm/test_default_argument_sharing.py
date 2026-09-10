"""Regression tests for the mutable-default-argument antipattern.

Several classic algorithm constructors used a class-object expression (built once,
at import time) as a default value for `selection`/`termination_criterion`. Every
instance built without passing that argument explicitly then shared the same
underlying object. This file guards against both instances of the bug being
reintroduced: selection_operator sharing (no live behavioral bug today, but blocks
per-instance rng injection) and termination_criterion sharing (a confirmed
correctness bug -- see the Fase 0 RNG-reproducibility plan).
"""

from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.moead import MOEAD, MOEADIEpsilon
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, Sphere
from jmetal.util.aggregation_function import WeightedSum
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9


def _zdt1_operators():
    problem = ZDT1()
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    return problem, mutation, crossover


class TestSelectionOperatorIsNotSharedAcrossInstances:
    def test_nsgaii_instances_built_without_an_explicit_selection_get_distinct_operators(self):
        problem, mutation, crossover = _zdt1_operators()

        algorithm_a = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=mutation,
            crossover=crossover,
        )
        algorithm_b = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=mutation,
            crossover=crossover,
        )

        assert algorithm_a.selection_operator is not algorithm_b.selection_operator

    def test_mocell_instances_built_without_an_explicit_selection_get_distinct_operators(self):
        problem, mutation, crossover = _zdt1_operators()

        def build():
            return MOCell(
                problem=problem,
                population_size=16,
                neighborhood=C9(rows=4, columns=4),
                archive=CrowdingDistanceArchive(16),
                mutation=mutation,
                crossover=crossover,
            )

        algorithm_a = build()
        algorithm_b = build()

        assert algorithm_a.selection_operator is not algorithm_b.selection_operator

    def test_genetic_algorithm_instances_built_without_an_explicit_selection_get_distinct_operators(
        self,
    ):
        problem = Sphere(number_of_variables=5)
        mutation = PolynomialMutation(probability=0.1, distribution_index=20)
        crossover = SBXCrossover(probability=0.9, distribution_index=20)

        def build():
            return GeneticAlgorithm(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
            )

        algorithm_a = build()
        algorithm_b = build()

        assert algorithm_a.selection_operator is not algorithm_b.selection_operator


def _moead_instance(problem=None, **overrides) -> MOEAD:
    problem = problem or ZDT1()
    kwargs = dict(
        problem=problem,
        population_size=5,
        mutation=PolynomialMutation(probability=0.0, distribution_index=20),
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        aggregation_function=WeightedSum(),
        neighbourhood_selection_probability=1.0,
        max_number_of_replaced_solutions=2,
        neighbor_size=2,
        weight_files_path="unused-for-2-objectives",
    )
    kwargs.update(overrides)
    return MOEAD(**kwargs)


class TestTerminationCriterionIsNotSharedAcrossInstances:
    """Guards the confirmed correctness bug: two instances of the same algorithm
    class built without an explicit termination_criterion used to share the same
    StoppingByEvaluations object, so a second instance could be born already
    "finished" if a prior instance of the same class had already run to completion
    in the same process.
    """

    def _assert_independent(self, algorithm_a, algorithm_b):
        assert algorithm_a.termination_criterion is not algorithm_b.termination_criterion

        algorithm_a.termination_criterion.update(EVALUATIONS=10**9)
        assert algorithm_a.termination_criterion.is_met
        assert not algorithm_b.termination_criterion.is_met

    def test_nsgaii_instances_get_independent_termination_criteria(self):
        problem, mutation, crossover = _zdt1_operators()

        def build():
            return NSGAII(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
            )

        self._assert_independent(build(), build())

    def test_genetic_algorithm_instances_get_independent_termination_criteria(self):
        problem = Sphere(number_of_variables=5)
        mutation = PolynomialMutation(probability=0.1, distribution_index=20)
        crossover = SBXCrossover(probability=0.9, distribution_index=20)

        def build():
            return GeneticAlgorithm(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
            )

        self._assert_independent(build(), build())

    def test_smsemoa_instances_get_independent_termination_criteria(self):
        problem, mutation, crossover = _zdt1_operators()

        def build():
            return SMSEMOA(
                problem=problem,
                population_size=10,
                mutation=mutation,
                crossover=crossover,
            )

        self._assert_independent(build(), build())

    def test_moead_instances_get_independent_termination_criteria(self):
        self._assert_independent(_moead_instance(), _moead_instance())

    def test_moead_i_epsilon_instances_get_independent_termination_criteria(self):
        def build():
            problem = ZDT1()
            return MOEADIEpsilon(
                problem=problem,
                population_size=5,
                mutation=PolynomialMutation(probability=0.0, distribution_index=20),
                crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
                aggregation_function=WeightedSum(),
                neighbourhood_selection_probability=1.0,
                max_number_of_replaced_solutions=2,
                neighbor_size=2,
                weight_files_path="unused-for-2-objectives",
            )

        algorithm_a, algorithm_b = build(), build()
        self._assert_independent(algorithm_a, algorithm_b)

        # MOEADIEpsilon keeps its own distinct default (300000, not the general
        # 25000) -- guard that the resolved default wasn't silently swapped.
        assert algorithm_a.termination_criterion.max_evaluations == 300000
