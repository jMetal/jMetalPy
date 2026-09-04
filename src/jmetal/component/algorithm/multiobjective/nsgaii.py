"""Factory for a component-based NSGA-II."""

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.catalogue.common.evaluation import SequentialEvaluation
from jmetal.component.catalogue.common.solutions_creation import RandomSolutionsCreation
from jmetal.component.catalogue.common.termination import Termination, TerminationByEvaluations
from jmetal.component.catalogue.ea.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
    Replacement,
)
from jmetal.component.catalogue.ea.selection import Selection, TournamentSelection
from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation, Variation
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.operator.selection import TournamentSelection as TournamentSelectionOperator
from jmetal.util.comparator import MultiComparator
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator
from jmetal.util.ranking import FastNonDominatedRanking

_DEFAULT_MAX_EVALUATIONS = 25000
_DEFAULT_TOURNAMENT_SIZE = 2


def build_nsgaii(
    problem: Problem,
    population_size: int,
    offspring_population_size: int,
    crossover: Crossover,
    mutation: Mutation,
    *,
    selection: Selection | None = None,
    variation: Variation | None = None,
    replacement: Replacement | None = None,
    termination: Termination | None = None,
    rng: np.random.Generator | None = None,
) -> EvolutionaryAlgorithm:
    """Build a component-based NSGA-II.

    A factory function, not a chainable builder class: `NSGAIIBuilder.java`'s
    `.setX().build()` pattern exists to simulate keyword arguments, which Python
    already has natively. Every parameter after `mutation` has NSGA-II's standard
    default and can be overridden individually.

    Args:
        problem: The problem to solve.
        population_size: The number of solutions kept across generations.
        offspring_population_size: The number of solutions produced each generation.
        crossover: The crossover operator.
        mutation: The mutation operator.
        selection: The mating selection strategy. Defaults to binary tournament
            using the same ranking and crowding-distance comparators as `replacement`.
        variation: The variation strategy. Defaults to `CrossoverAndMutationVariation`
            with `crossover` and `mutation`.
        replacement: The replacement strategy. Defaults to
            `RankingAndDensityEstimatorReplacement` with fast non-dominated ranking,
            crowding distance, and `RemovalPolicyType.ONE_SHOT` -- NSGA-II's standard
            configuration.
        termination: The termination condition. Defaults to 25000 evaluations.
        rng: The random generator shared with every RNG-aware component. Defaults to
            a fresh `np.random.default_rng()`.

    Returns:
        An `EvolutionaryAlgorithm` configured as NSGA-II, ready to `.run()`.
    """
    ranking = FastNonDominatedRanking()
    density_estimator = CrowdingDistanceDensityEstimator()

    if variation is None:
        variation = CrossoverAndMutationVariation(offspring_population_size, crossover, mutation)

    if replacement is None:
        replacement = RankingAndDensityEstimatorReplacement(
            ranking, density_estimator, RemovalPolicyType.ONE_SHOT
        )

    if selection is None:
        comparator = MultiComparator(
            [FastNonDominatedRanking.get_comparator(), CrowdingDistanceDensityEstimator.get_comparator()]
        )
        selection = TournamentSelection(
            TournamentSelectionOperator(_DEFAULT_TOURNAMENT_SIZE, comparator),
            mating_pool_size=variation.mating_pool_size(),
        )

    if termination is None:
        termination = TerminationByEvaluations(max_evaluations=_DEFAULT_MAX_EVALUATIONS)

    return EvolutionaryAlgorithm(
        name="NSGAII",
        solutions_creation=RandomSolutionsCreation(problem, population_size),
        evaluation=SequentialEvaluation(problem),
        termination=termination,
        selection=selection,
        variation=variation,
        replacement=replacement,
        rng=rng,
    )
