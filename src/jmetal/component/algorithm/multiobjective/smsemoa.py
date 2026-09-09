"""Factory for a component-based SMS-EMOA."""

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.catalogue.common.evaluation import (
    SequentialEvaluation,
    SequentialEvaluationWithArchive,
)
from jmetal.component.catalogue.common.solutions_creation import RandomSolutionsCreation
from jmetal.component.catalogue.common.termination import Termination, TerminationByEvaluations
from jmetal.component.catalogue.ea.replacement import Replacement, SMSEMOAReplacement
from jmetal.component.catalogue.ea.selection import RandomSelection, Selection
from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation, Variation
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.operator.selection import RandomSelection as RandomSelectionOperator
from jmetal.util.archive import Archive

_DEFAULT_MAX_EVALUATIONS = 25000
_STEADY_STATE_OFFSPRING_SIZE = 1


def build_smsemoa(
    problem: Problem,
    population_size: int,
    crossover: Crossover,
    mutation: Mutation,
    *,
    selection: Selection | None = None,
    variation: Variation | None = None,
    replacement: Replacement | None = None,
    termination: Termination | None = None,
    rng: np.random.Generator | None = None,
    archive: Archive | None = None,
) -> EvolutionaryAlgorithm:
    """Build a component-based SMS-EMOA.

    SMS-EMOA is very similar to NSGA-II: same six-component template, same factory
    pattern as `build_nsgaii()`. It differs only in mating selection (uniformly random
    rather than tournament) and replacement (hypervolume-contribution-based pruning of
    the worst non-dominated front rather than ranking plus crowding distance).

    Unlike `build_nsgaii()`, there is no `offspring_population_size` parameter: SMS-EMOA
    is steady-state by definition (Beume et al., 2007) and always produces exactly one
    offspring per generation, matching
    `jmetal.algorithm.multiobjective.smsemoa.SMSEMOA`, which hardcodes the same value.

    Args:
        problem: The problem to solve.
        population_size: The number of solutions kept across generations.
        crossover: The crossover operator.
        mutation: The mutation operator.
        selection: The mating selection strategy. Defaults to `RandomSelection`.
        variation: The variation strategy. Defaults to `CrossoverAndMutationVariation`
            with `crossover`, `mutation`, and an offspring population size of 1.
        replacement: The replacement strategy. Defaults to `SMSEMOAReplacement`, which
            recomputes its hypervolume reference point from the merged population on
            every call -- SMS-EMOA's standard configuration.
        termination: The termination condition. Defaults to 25000 evaluations.
        rng: The random generator shared with every RNG-aware component. Defaults to
            a fresh `np.random.default_rng()`.
        archive: An optional external archive (e.g. `NonDominatedSolutionsArchive`
            for an unbounded archive, or `CrowdingDistanceArchive(size)` for a
            bounded one). When given, every evaluated solution is also copied into
            it, and `result()` returns the archive's contents instead of the final
            population -- the population still drives selection/replacement as
            usual, so the archive is a pure addition, not a change of algorithm.

    Returns:
        An `EvolutionaryAlgorithm` configured as SMS-EMOA, ready to `.run()`.
    """
    if variation is None:
        variation = CrossoverAndMutationVariation(_STEADY_STATE_OFFSPRING_SIZE, crossover, mutation)

    if replacement is None:
        replacement = SMSEMOAReplacement()

    if selection is None:
        selection = RandomSelection(
            RandomSelectionOperator(), mating_pool_size=variation.mating_pool_size()
        )

    if termination is None:
        termination = TerminationByEvaluations(max_evaluations=_DEFAULT_MAX_EVALUATIONS)

    if archive is not None:
        evaluation = SequentialEvaluationWithArchive(problem, archive)
    else:
        evaluation = SequentialEvaluation(problem)

    return EvolutionaryAlgorithm(
        name="SMSEMOA",
        solutions_creation=RandomSolutionsCreation(problem, population_size, rng=rng),
        evaluation=evaluation,
        termination=termination,
        selection=selection,
        variation=variation,
        replacement=replacement,
        rng=rng,
        archive=archive,
    )
