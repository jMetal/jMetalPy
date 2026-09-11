"""Factories for a component-based MOEA/D: classic (any crossover) and MOEA/D-DE."""

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from jmetal.component.catalogue.common.evaluation import (
    SequentialEvaluation,
    SequentialEvaluationWithArchive,
)
from jmetal.component.catalogue.common.solutions_creation import RandomSolutionsCreation
from jmetal.component.catalogue.common.termination import Termination, TerminationByEvaluations
from jmetal.component.catalogue.ea.moead import (
    DifferentialEvolutionCrossoverVariation,
    MOEADContext,
    MOEADReplacement,
    MOEADSelection,
    PermutationCycle,
    SubproblemSequenceGenerator,
)
from jmetal.component.catalogue.ea.replacement import Replacement
from jmetal.component.catalogue.ea.selection import Selection
from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation, Variation
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.util.aggregation_function import (
    AggregationFunction,
    PenaltyBoundaryIntersection,
    Tschebycheff,
)
from jmetal.util.archive import Archive
from jmetal.util.neighborhood import WeightVectorNeighborhood

_DEFAULT_MAX_EVALUATIONS = 25000
_DEFAULT_NEIGHBOURHOOD_SIZE = 20
_DEFAULT_NEIGHBOURHOOD_SELECTION_PROBABILITY = 0.9
_DEFAULT_MAX_NUMBER_OF_REPLACED_SOLUTIONS = 2
_DEFAULT_WEIGHT_FILES_PATH = "resources/MOEAD_weights"
_NUMBER_OF_MATING_PARENTS = 2


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Resolve `rng` once, up front. `MOEADContext` needs a concrete generator before
    it can be constructed, and it must be constructed before `EvolutionaryAlgorithm`
    (its `Selection`/`Variation`/`Replacement` depend on it) -- so, unlike
    `build_nsgaii()`/`build_smsemoa()`, this can't be left to
    `EvolutionaryAlgorithm.__init__`'s own `rng or np.random.default_rng()` fallback.
    Both factories below pass this same resolved instance to `MOEADContext` and to
    `EvolutionaryAlgorithm(rng=...)`, so they always share one generator.
    """
    return rng if rng is not None else np.random.default_rng()


def _build_context(
    population_size: int,
    neighbourhood_selection_probability: float,
    subproblem_id_generator: SubproblemSequenceGenerator | None,
    rng: np.random.Generator,
) -> MOEADContext:
    if subproblem_id_generator is None:
        subproblem_id_generator = PermutationCycle(population_size, rng=rng)
    return MOEADContext(subproblem_id_generator, neighbourhood_selection_probability, rng=rng)


def build_moead(
    problem: Problem,
    population_size: int,
    crossover: Crossover,
    mutation: Mutation,
    *,
    aggregation_function: AggregationFunction | None = None,
    neighbourhood_size: int = _DEFAULT_NEIGHBOURHOOD_SIZE,
    neighbourhood_selection_probability: float = _DEFAULT_NEIGHBOURHOOD_SELECTION_PROBABILITY,
    max_number_of_replaced_solutions: int = _DEFAULT_MAX_NUMBER_OF_REPLACED_SOLUTIONS,
    weight_files_path: str = _DEFAULT_WEIGHT_FILES_PATH,
    subproblem_id_generator: SubproblemSequenceGenerator | None = None,
    selection: Selection | None = None,
    variation: Variation | None = None,
    replacement: Replacement | None = None,
    termination: Termination | None = None,
    rng: np.random.Generator | None = None,
    archive: Archive | None = None,
) -> EvolutionaryAlgorithm:
    """Build a component-based, classic MOEA/D -- any crossover/mutation pair (SBX and
    polynomial mutation are the usual choice), not specifically differential evolution.
    For MOEA/D-DE, see `build_moead_de()`.

    Unlike jMetalPy's existing `jmetal.algorithm.multiobjective.moead.MOEAD` (which,
    despite its name, is *already* MOEA/D-DE -- its `crossover` parameter is typed as
    `DifferentialEvolutionCrossover` and required), this accepts any `Crossover`, since
    the `Variation` it defaults to (`CrossoverAndMutationVariation`) doesn't need DE's
    "current subproblem as a third parent" mechanism the way `build_moead_de()` does.

    Args:
        problem: The problem to solve.
        population_size: The number of subproblems (one weight vector, and one
            population slot, per subproblem).
        crossover: The crossover operator.
        mutation: The mutation operator.
        aggregation_function: Scalarizes an objective vector against a weight vector.
            Defaults to `PenaltyBoundaryIntersection` (`theta=5.0`) -- MOEA/D Java's
            `MOEADBuilder` default.
        neighbourhood_size: How many of the closest weight vectors form each
            subproblem's neighborhood.
        neighbourhood_selection_probability: Probability of building the mating pool,
            and later scanning for replacement, from the current subproblem's
            neighborhood rather than the whole population (`Delta` in Zhang & Li's
            paper).
        max_number_of_replaced_solutions: Caps how many population slots one offspring
            can replace (`eta` in Zhang & Li's paper).
        weight_files_path: Directory holding precomputed weight-vector files (used for
            3+ objectives; 2-objective weights are generated analytically and don't
            need one). Defaults to this repository's bundled
            `resources/MOEAD_weights`.
        subproblem_id_generator: Decides which subproblem each iteration processes.
            Defaults to `PermutationCycle` (a random, non-repeating order, reshuffled
            once exhausted -- matching the classic algorithm's own strategy).
        selection: The mating selection strategy. Defaults to `MOEADSelection`.
        variation: The variation strategy. Defaults to `CrossoverAndMutationVariation`
            with `crossover`, `mutation` and an offspring population size of 1 (MOEA/D
            is steady-state by definition).
        replacement: The replacement strategy. Defaults to `MOEADReplacement`.
        termination: The termination condition. Defaults to 25000 evaluations.
        rng: The random generator shared by every MOEA/D-specific random decision
            (which subproblem, which scope, mating-pool sampling, the replacement
            scan) and by the initial population. Defaults to a fresh
            `np.random.default_rng()`. Nothing this factory constructs itself ever
            falls back to the global `random`/`numpy.random` state, unlike
            `build_nsgaii()`'s `TournamentSelection` operator (which has no `rng`
            parameter at all, so full reproducibility isn't achievable for it no
            matter what's passed in). For MOEA/D, passing the same `rng` here *and*
            to `crossover`/`mutation` (both are your own operators, constructed
            outside this factory, so their reproducibility is in your hands the same
            way it already is for `build_nsgaii()`/`build_smsemoa()`) reproduces an
            entire run bit-for-bit from a single seed.
        archive: An optional external archive, identical in effect to
            `build_nsgaii()`'s/`build_smsemoa()`'s.

    Returns:
        An `EvolutionaryAlgorithm` configured as MOEA/D, ready to `.run()`.
    """
    rng = _resolve_rng(rng)

    neighbourhood = WeightVectorNeighborhood(
        number_of_weight_vectors=population_size,
        neighborhood_size=neighbourhood_size,
        weight_vector_size=problem.number_of_objectives(),
        weights_path=weight_files_path,
    )
    context = _build_context(
        population_size, neighbourhood_selection_probability, subproblem_id_generator, rng
    )

    if aggregation_function is None:
        aggregation_function = PenaltyBoundaryIntersection(dimension=problem.number_of_objectives())

    if variation is None:
        variation = CrossoverAndMutationVariation(1, crossover, mutation)

    if selection is None:
        selection = MOEADSelection(context, neighbourhood, number_of_parents=_NUMBER_OF_MATING_PARENTS)

    if replacement is None:
        replacement = MOEADReplacement(
            context, neighbourhood, aggregation_function, max_number_of_replaced_solutions
        )

    if termination is None:
        termination = TerminationByEvaluations(max_evaluations=_DEFAULT_MAX_EVALUATIONS)

    if archive is not None:
        evaluation = SequentialEvaluationWithArchive(problem, archive)
    else:
        evaluation = SequentialEvaluation(problem)

    return EvolutionaryAlgorithm(
        name="MOEAD",
        solutions_creation=RandomSolutionsCreation(problem, population_size, rng=rng),
        evaluation=evaluation,
        termination=termination,
        selection=selection,
        variation=variation,
        replacement=replacement,
        rng=rng,
        archive=archive,
    )


def build_moead_de(
    problem: Problem,
    population_size: int,
    mutation: Mutation,
    *,
    cr: float = 1.0,
    f: float = 0.5,
    aggregation_function: AggregationFunction | None = None,
    neighbourhood_size: int = _DEFAULT_NEIGHBOURHOOD_SIZE,
    neighbourhood_selection_probability: float = _DEFAULT_NEIGHBOURHOOD_SELECTION_PROBABILITY,
    max_number_of_replaced_solutions: int = _DEFAULT_MAX_NUMBER_OF_REPLACED_SOLUTIONS,
    weight_files_path: str = _DEFAULT_WEIGHT_FILES_PATH,
    subproblem_id_generator: SubproblemSequenceGenerator | None = None,
    selection: Selection | None = None,
    variation: Variation | None = None,
    replacement: Replacement | None = None,
    termination: Termination | None = None,
    rng: np.random.Generator | None = None,
    archive: Archive | None = None,
) -> EvolutionaryAlgorithm:
    """Build a component-based MOEA/D-DE -- differential-evolution crossover in place
    of a generic crossover operator. For the classic (any-crossover) variant, see
    `build_moead()`.

    This is the variant jMetalPy's existing
    `jmetal.algorithm.multiobjective.moead.MOEAD` actually implements (despite its
    plain name -- see `build_moead()`'s docstring).

    Args:
        problem: The problem to solve.
        population_size: The number of subproblems (one weight vector, and one
            population slot, per subproblem).
        mutation: The mutation operator, applied to the single DE offspring.
        cr: The DE crossover probability (`CR`).
        f: The DE differential weight (`F`).
        aggregation_function: Scalarizes an objective vector against a weight vector.
            Defaults to `Tschebycheff` -- MOEA/D Java's `MOEADDEBuilder` default.
        neighbourhood_size: How many of the closest weight vectors form each
            subproblem's neighborhood.
        neighbourhood_selection_probability: Probability of building the mating pool,
            and later scanning for replacement, from the current subproblem's
            neighborhood rather than the whole population (`Delta` in Zhang & Li's
            paper).
        max_number_of_replaced_solutions: Caps how many population slots one offspring
            can replace (`eta` in Zhang & Li's paper).
        weight_files_path: Directory holding precomputed weight-vector files (used for
            3+ objectives; 2-objective weights are generated analytically and don't
            need one). Defaults to this repository's bundled
            `resources/MOEAD_weights`.
        subproblem_id_generator: Decides which subproblem each iteration processes.
            Defaults to `PermutationCycle`.
        selection: The mating selection strategy. Defaults to `MOEADSelection`.
        variation: The variation strategy. Defaults to
            `DifferentialEvolutionCrossoverVariation` with a fresh
            `DifferentialEvolutionCrossover(cr, f, rng=rng)` and `mutation`.
        replacement: The replacement strategy. Defaults to `MOEADReplacement`.
        termination: The termination condition. Defaults to 25000 evaluations.
        rng: The random generator shared by every MOEA/D-specific random decision, the
            DE crossover operator this factory constructs from `cr`/`f` (unlike
            `build_moead()`'s `crossover`, which is your own, so this one *is*
            covered automatically), and the initial population. Defaults to a fresh
            `np.random.default_rng()`. Passing the same `rng` here *and* to
            `mutation` (your own operator, same caveat as `build_moead()`'s
            `crossover`/`mutation`) reproduces an entire run bit-for-bit from a
            single seed (see `build_moead()`'s docstring).
        archive: An optional external archive, identical in effect to
            `build_nsgaii()`'s/`build_smsemoa()`'s.

    Returns:
        An `EvolutionaryAlgorithm` configured as MOEA/D-DE, ready to `.run()`.
    """
    rng = _resolve_rng(rng)

    neighbourhood = WeightVectorNeighborhood(
        number_of_weight_vectors=population_size,
        neighborhood_size=neighbourhood_size,
        weight_vector_size=problem.number_of_objectives(),
        weights_path=weight_files_path,
    )
    context = _build_context(
        population_size, neighbourhood_selection_probability, subproblem_id_generator, rng
    )

    if aggregation_function is None:
        aggregation_function = Tschebycheff(dimension=problem.number_of_objectives())

    if variation is None:
        crossover = DifferentialEvolutionCrossover(CR=cr, F=f, rng=rng)
        variation = DifferentialEvolutionCrossoverVariation(context, crossover, mutation)

    if selection is None:
        selection = MOEADSelection(context, neighbourhood, number_of_parents=_NUMBER_OF_MATING_PARENTS)

    if replacement is None:
        replacement = MOEADReplacement(
            context, neighbourhood, aggregation_function, max_number_of_replaced_solutions
        )

    if termination is None:
        termination = TerminationByEvaluations(max_evaluations=_DEFAULT_MAX_EVALUATIONS)

    if archive is not None:
        evaluation = SequentialEvaluationWithArchive(problem, archive)
    else:
        evaluation = SequentialEvaluation(problem)

    return EvolutionaryAlgorithm(
        name="MOEAD-DE",
        solutions_creation=RandomSolutionsCreation(problem, population_size, rng=rng),
        evaluation=evaluation,
        termination=termination,
        selection=selection,
        variation=variation,
        replacement=replacement,
        rng=rng,
        archive=archive,
    )
