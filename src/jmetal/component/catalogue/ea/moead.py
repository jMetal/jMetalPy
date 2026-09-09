"""MOEA/D-specific components.

MOEA/D processes one subproblem per iteration: pick a subproblem, build a mating pool
scoped to its neighborhood (or the whole population), produce one offspring, and
compare it against neighboring (or every) subproblem's weight vector to decide which
solutions it replaces. None of that fits cleanly into `Selection.select()`,
`Variation.variate()` and `Replacement.replace()`'s signatures alone: none of them
carries a "which subproblem is this iteration about" concept, and none should --
that's specific to MOEA/D, not a general evolutionary-algorithm concept the six-component
template should know about.

jMetal Java's component-based MOEA/D (`MOEADBuilder`/`MOEADDEBuilder` in
`jmetal-component`) solves the same problem the same way: a `SequenceGenerator<Integer>`
constructed once and passed by reference into `Selection`, `Variation` and
`Replacement`, each of which consults it without the generic `EvolutionaryAlgorithm`
template needing to change. `MOEADContext` plays that role here, and additionally
carries the shared `rng` and the neighborhood-vs-population scope decision for the
current iteration (jMetal Java threads those through other means not ported here).

All randomness specific to MOEA/D -- which subproblem, which scope, which mating-pool
members, which permutation to scan during replacement -- is drawn exclusively from
`MOEADContext.rng`, never from the global `random`/`numpy.random` state. This is a
deliberate departure from the classic `jmetal.algorithm.multiobjective.moead.MOEAD`,
which mixes three incompatible random sources (global `random`, global legacy
`numpy.random`, and each operator's own `np.random.Generator`) and can therefore never
be made fully reproducible from a single seed. See `MODERNIZATION.md` for the trade-off
this implies: no execution-level equivalence test against the classic algorithm is
possible, only a structural one (`MOEADReplacement` given identical inputs) plus
quality-indicator-floor integration tests.
"""

import copy
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from jmetal.core.operator import Mutation
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.replacement import Replacement
from jmetal.operator.selection import NaryRandomSolutionSelection
from jmetal.util.aggregation_function import AggregationFunction
from jmetal.util.neighborhood import WeightVectorNeighborhood

S = TypeVar("S")


@runtime_checkable
class SubproblemSequenceGenerator(Protocol):
    """Hands out subproblem indices, one per iteration.

    `get_value()` always returns the index for the *current* iteration; a fresh
    instance must already have a valid value before `generate_next_value()` is ever
    called, so the first `get_value()` of a run doesn't require an extra advance.
    """

    def get_value(self) -> int:
        """Return the subproblem index for the current iteration."""
        ...

    def generate_next_value(self) -> None:
        """Advance to the next iteration's subproblem index."""
        ...


class PermutationCycle:
    """Hands out indices `[0, size)` in a random order, without repeating until the
    permutation is exhausted, at which point it reshuffles -- the same idea as the
    classic `MOEAD`'s `Permutation` helper
    (`jmetal.algorithm.multiobjective.moead.Permutation`), but drawing from the shared
    `rng: np.random.Generator` instead of the global legacy `numpy.random` state the
    original uses, so a run built from this stays reproducible from a single seed.

    Args:
        size: The number of subproblems (one index per subproblem, i.e.
            `population_size`).
        rng: The shared random generator.
    """

    def __init__(self, size: int, rng: np.random.Generator):
        self.size = size
        self.rng = rng
        self._permutation = self.rng.permutation(size).tolist()
        self._index = 0

    def get_value(self) -> int:
        return self._permutation[self._index]

    def generate_next_value(self) -> None:
        self._index += 1
        if self._index == self.size:
            self._permutation = self.rng.permutation(self.size).tolist()
            self._index = 0


class CyclicSequence:
    """Hands out indices `[0, size)` in a fixed, repeating order -- the deterministic
    alternative Evolver's MOEA/D parameter space exposes as
    `subProblemIdGenerator: cyclicIntegerSequence`. Consumes no randomness.

    Args:
        size: The number of subproblems.
    """

    def __init__(self, size: int):
        self.size = size
        self._current_value = 0

    def get_value(self) -> int:
        return self._current_value

    def generate_next_value(self) -> None:
        self._current_value = (self._current_value + 1) % self.size


class MOEADContext:
    """Shared per-iteration state, consulted (never independently recomputed) by
    `MOEADSelection`, `DifferentialEvolutionCrossoverVariation` and
    `MOEADReplacement` in the same iteration. See the module docstring for why this
    exists instead of extending the generic `Selection`/`Variation`/`Replacement`
    protocols.

    Args:
        sequence_generator: Decides which subproblem each iteration processes.
        neighbourhood_selection_probability: Probability of building the mating pool
            (and later, the replacement scan) from the current subproblem's
            neighborhood rather than the whole population (`Delta` in Zhang & Li's
            paper).
        rng: The shared random generator -- the only source of randomness for
            everything MOEA/D-specific (see the module docstring).
    """

    def __init__(
        self,
        sequence_generator: SubproblemSequenceGenerator,
        neighbourhood_selection_probability: float,
        rng: np.random.Generator,
    ):
        self.sequence_generator = sequence_generator
        self.neighbourhood_selection_probability = neighbourhood_selection_probability
        self.rng = rng
        self.current_subproblem_id: int = sequence_generator.get_value()
        self.neighbor_type: str = "NEIGHBOR"

    def advance(self) -> None:
        """Move to the next iteration: read this iteration's subproblem id, decide
        its neighborhood-vs-population scope, and advance the sequence generator for
        next time. Called once per iteration, only by `MOEADSelection`."""
        self.current_subproblem_id = self.sequence_generator.get_value()
        self.neighbor_type = (
            "NEIGHBOR"
            if self.rng.random() < self.neighbourhood_selection_probability
            else "POPULATION"
        )
        self.sequence_generator.generate_next_value()


class MOEADSelection(Generic[S]):
    """Builds the mating pool for the current subproblem: `number_of_parents`
    solutions drawn from its neighborhood or from the whole population, according to
    `context.neighbor_type` (decided by `context.advance()`, called here at the start
    of `select()`).

    Args:
        context: The shared `MOEADContext` for this run.
        neighbourhood: The weight-vector neighborhood structure.
        number_of_parents: How many solutions to sample -- 2 for both the classic
            (SBX) and DE variants (DE's third "parent", the current subproblem's own
            solution, is fetched directly from the population by
            `DifferentialEvolutionCrossoverVariation`, not sampled here).
        selection_operator: The sampling operator. Defaults to
            `NaryRandomSolutionSelection(number_of_parents, rng=context.rng)`.
    """

    def __init__(
        self,
        context: MOEADContext,
        neighbourhood: WeightVectorNeighborhood,
        number_of_parents: int,
        selection_operator: NaryRandomSolutionSelection | None = None,
    ):
        self.context = context
        self.neighbourhood = neighbourhood
        self.selection_operator = (
            selection_operator
            if selection_operator is not None
            else NaryRandomSolutionSelection(number_of_parents, rng=context.rng)
        )

    def select(self, solution_list: list[S]) -> list[S]:
        """Advance to the next subproblem and sample its mating pool.

        Args:
            solution_list: The current population.

        Returns:
            The mating pool for this iteration's subproblem.
        """
        self.context.advance()
        pool = (
            self.neighbourhood.get_neighbors(self.context.current_subproblem_id, solution_list)
            if self.context.neighbor_type == "NEIGHBOR"
            else solution_list
        )
        return self.selection_operator.execute(pool)


class DifferentialEvolutionCrossoverVariation(Generic[S]):
    """Produces one offspring via differential-evolution crossover, for the MOEA/D-DE
    variant. Directly analogous to jmetal-component's
    `DifferentialEvolutionCrossoverVariation.java`: the DE target/current vector is
    fetched by indexing `solution_list` (the full population, already passed to
    `variate()`) at `context.current_subproblem_id`, rather than being part of the
    mating pool `MOEADSelection` returns -- no protocol change needed.

    Args:
        context: The shared `MOEADContext` for this run.
        crossover: The differential-evolution crossover operator (needs exactly 3
            parents: the mating pool's 2 plus the current subproblem's own solution).
        mutation: Applied to the single resulting offspring.
    """

    def __init__(
        self,
        context: MOEADContext,
        crossover: DifferentialEvolutionCrossover,
        mutation: Mutation,
    ):
        self.context = context
        self.crossover = crossover
        self.mutation = mutation

    def variate(self, solution_list: list[S], mating_pool: list[S]) -> list[S]:
        """Run DE crossover (2 mating-pool parents + the subproblem's own solution)
        then mutate the single resulting offspring.

        Args:
            solution_list: The current population.
            mating_pool: The 2 solutions `MOEADSelection` sampled for this iteration.

        Returns:
            A single-element offspring list.
        """
        current_solution = solution_list[self.context.current_subproblem_id]
        self.crossover.current_individual = current_solution
        offspring_population = self.crossover.execute([*mating_pool, current_solution])
        self.mutation.execute(offspring_population[0])
        return offspring_population

    def mating_pool_size(self) -> int:
        """Return how many parents `variate()` expects in the mating pool."""
        return 2

    def offspring_population_size(self) -> int:
        """Return how many offspring `variate()` produces."""
        return 1


class MOEADReplacement(Replacement[S]):
    """Updates the population after one subproblem's offspring is produced:
    generalizes the classic `MOEAD.update_current_subproblem_neighborhood()`
    (`jmetal.algorithm.multiobjective.moead`), reading `context.current_subproblem_id`
    and `context.neighbor_type` (fixed by `MOEADSelection` earlier in the same
    iteration) instead of instance attributes shared by being methods of the same
    object.

    The scan order matters and is asymmetric, matching the classic algorithm exactly:
    in `NEIGHBOR` scope, the subproblem's neighbors are scanned in the order
    `WeightVectorNeighborhood` precomputed them (closest weight vector first) --
    *not* shuffled. Only `POPULATION` scope draws a fresh permutation (via
    `context.rng`, not the classic algorithm's global legacy `numpy.random`).

    Args:
        context: The shared `MOEADContext` for this run.
        neighbourhood: The weight-vector neighborhood structure.
        aggregation_function: Scalarizes an objective vector against a weight vector;
            its ideal point is updated here with every new offspring.
        max_number_of_replaced_solutions: Caps how many population slots one
            offspring can replace (`eta` in Zhang & Li's paper).
    """

    def __init__(
        self,
        context: MOEADContext,
        neighbourhood: WeightVectorNeighborhood,
        aggregation_function: AggregationFunction,
        max_number_of_replaced_solutions: int,
    ):
        self.context = context
        self.neighbourhood = neighbourhood
        self.aggregation_function = aggregation_function
        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions

    def replace(self, solution_list: list[S], offspring_list: list[S]) -> list[S]:
        """Update the ideal point, then replace up to
        `max_number_of_replaced_solutions` population slots whose scalarized value
        the new offspring improves on, scanned in `context.neighbor_type`'s scope.

        Args:
            solution_list: The current population (updated, and returned, in place).
            offspring_list: The single offspring produced this iteration.

        Returns:
            `solution_list`, with up to `max_number_of_replaced_solutions` slots
            replaced by a copy of the new offspring.
        """
        new_solution = offspring_list[0]
        self.aggregation_function.update(new_solution.objectives)

        if self.context.neighbor_type == "NEIGHBOR":
            candidate_indexes = self.neighbourhood.get_neighborhood()[
                self.context.current_subproblem_id
            ].tolist()
        else:
            candidate_indexes = self.context.rng.permutation(len(solution_list)).tolist()

        replacements = 0
        for k in candidate_indexes:
            weight_vector = self.neighbourhood.weight_vectors[k]
            new_value = self.aggregation_function.compute(new_solution.objectives, weight_vector)
            current_value = self.aggregation_function.compute(
                solution_list[k].objectives, weight_vector
            )
            if new_value < current_value:
                solution_list[k] = copy.copy(new_solution)
                replacements += 1
            if replacements >= self.max_number_of_replaced_solutions:
                break

        return solution_list
