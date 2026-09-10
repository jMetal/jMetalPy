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
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, Sphere
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
