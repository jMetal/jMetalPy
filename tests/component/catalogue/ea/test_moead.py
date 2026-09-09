"""Tests for the MOEA/D-specific components."""

import numpy as np
import pytest

from jmetal.component.catalogue.ea.moead import (
    CyclicSequence,
    DifferentialEvolutionCrossoverVariation,
    MOEADContext,
    MOEADReplacement,
    MOEADSelection,
    PermutationCycle,
    SubproblemSequenceGenerator,
)
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.aggregation_function import Tschebycheff, WeightedSum
from jmetal.util.neighborhood import WeightVectorNeighborhood


def _solution(objectives: list[float]) -> FloatSolution:
    solution = FloatSolution(
        lower_bound=[0.0] * len(objectives),
        upper_bound=[1.0] * len(objectives),
        number_of_objectives=len(objectives),
    )
    solution.objectives = objectives
    solution.variables = [0.5] * len(objectives)
    return solution


class TestPermutationCycle:
    def test_yields_every_index_exactly_once_before_repeating(self):
        cycle = PermutationCycle(5, rng=np.random.default_rng(1))

        seen = []
        for _ in range(5):
            seen.append(cycle.get_value())
            cycle.generate_next_value()

        assert sorted(seen) == list(range(5))

    def test_reshuffles_after_exhausting_the_permutation(self):
        cycle = PermutationCycle(5, rng=np.random.default_rng(1))
        for _ in range(5):
            cycle.generate_next_value()

        # After a full cycle, get_value() must still be a valid index -- the
        # permutation was regenerated, not left exhausted/out of range.
        assert 0 <= cycle.get_value() < 5

    def test_deterministic_for_a_fixed_seed(self):
        cycle_a = PermutationCycle(10, rng=np.random.default_rng(42))
        cycle_b = PermutationCycle(10, rng=np.random.default_rng(42))

        sequence_a = []
        sequence_b = []
        for _ in range(25):
            sequence_a.append(cycle_a.get_value())
            cycle_a.generate_next_value()
            sequence_b.append(cycle_b.get_value())
            cycle_b.generate_next_value()

        assert sequence_a == sequence_b

    def test_satisfies_the_subproblem_sequence_generator_protocol(self):
        assert isinstance(PermutationCycle(3, rng=np.random.default_rng(1)), SubproblemSequenceGenerator)


class TestCyclicSequence:
    def test_cycles_through_indices_in_fixed_order(self):
        sequence = CyclicSequence(3)

        values = []
        for _ in range(7):
            values.append(sequence.get_value())
            sequence.generate_next_value()

        assert values == [0, 1, 2, 0, 1, 2, 0]

    def test_satisfies_the_subproblem_sequence_generator_protocol(self):
        assert isinstance(CyclicSequence(3), SubproblemSequenceGenerator)


class TestMOEADContext:
    def test_starts_at_the_sequence_generators_initial_value(self):
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=0.9, rng=np.random.default_rng(1)
        )

        assert context.current_subproblem_id == 0

    def test_advance_updates_the_current_subproblem_id(self):
        context = MOEADContext(
            CyclicSequence(3), neighbourhood_selection_probability=0.9, rng=np.random.default_rng(1)
        )

        context.advance()
        first = context.current_subproblem_id
        context.advance()
        second = context.current_subproblem_id

        assert (first, second) == (0, 1)

    def test_neighbor_type_is_always_neighbor_when_probability_is_one(self):
        context = MOEADContext(
            CyclicSequence(3), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )

        for _ in range(20):
            context.advance()
            assert context.neighbor_type == "NEIGHBOR"

    def test_neighbor_type_is_always_population_when_probability_is_zero(self):
        context = MOEADContext(
            CyclicSequence(3), neighbourhood_selection_probability=0.0, rng=np.random.default_rng(1)
        )

        for _ in range(20):
            context.advance()
            assert context.neighbor_type == "POPULATION"


class TestMOEADSelection:
    def _neighbourhood(self) -> WeightVectorNeighborhood:
        return WeightVectorNeighborhood(number_of_weight_vectors=5, neighborhood_size=2)

    def test_returns_a_mating_pool_of_the_requested_size(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        selection = MOEADSelection(context, self._neighbourhood(), number_of_parents=2)

        mating_pool = selection.select(population)

        assert len(mating_pool) == 2

    def test_advances_the_context_on_every_call(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        selection = MOEADSelection(context, self._neighbourhood(), number_of_parents=2)

        selection.select(population)
        first = context.current_subproblem_id
        selection.select(population)
        second = context.current_subproblem_id

        assert (first, second) == (0, 1)

    def test_selects_only_from_the_neighborhood_when_scope_is_neighbor(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        neighbourhood = self._neighbourhood()
        selection = MOEADSelection(context, neighbourhood, number_of_parents=2)

        mating_pool = selection.select(population)

        neighbor_indexes = neighbourhood.get_neighborhood()[0].tolist()
        allowed_ids = {id(population[i]) for i in neighbor_indexes}
        assert all(id(s) in allowed_ids for s in mating_pool)

    def test_selects_from_the_whole_population_when_scope_is_population(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=0.0, rng=np.random.default_rng(1)
        )
        selection = MOEADSelection(context, self._neighbourhood(), number_of_parents=2)

        mating_pool = selection.select(population)

        assert all(s in population for s in mating_pool)


class TestDifferentialEvolutionCrossoverVariation:
    def test_uses_the_current_subproblems_own_solution_as_the_de_target(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 2
        crossover = DifferentialEvolutionCrossover(CR=1.0, F=0.5, rng=np.random.default_rng(1))
        mutation = PolynomialMutation(probability=0.0, distribution_index=20, rng=np.random.default_rng(1))
        variation = DifferentialEvolutionCrossoverVariation(context, crossover, mutation)

        mating_pool = [population[0], population[1]]
        variation.variate(population, mating_pool)

        assert crossover.current_individual is population[2]

    def test_returns_a_single_offspring(self):
        population = [_solution([float(i), float(5 - i)]) for i in range(5)]
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        crossover = DifferentialEvolutionCrossover(CR=1.0, F=0.5, rng=np.random.default_rng(1))
        mutation = PolynomialMutation(probability=0.0, distribution_index=20, rng=np.random.default_rng(1))
        variation = DifferentialEvolutionCrossoverVariation(context, crossover, mutation)

        offspring = variation.variate(population, [population[0], population[1]])

        assert len(offspring) == 1

    def test_mating_pool_size_and_offspring_population_size(self):
        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        crossover = DifferentialEvolutionCrossover(CR=1.0, F=0.5, rng=np.random.default_rng(1))
        mutation = PolynomialMutation(probability=0.0, distribution_index=20, rng=np.random.default_rng(1))
        variation = DifferentialEvolutionCrossoverVariation(context, crossover, mutation)

        assert variation.mating_pool_size() == 2
        assert variation.offspring_population_size() == 1


class TestMOEADReplacement:
    def _neighbourhood(self) -> WeightVectorNeighborhood:
        return WeightVectorNeighborhood(number_of_weight_vectors=4, neighborhood_size=2)

    def test_replaces_a_solution_the_offspring_improves_on(self):
        population = [
            _solution([1.0, 0.0]),
            _solution([0.75, 0.25]),
            _solution([0.25, 0.75]),
            _solution([0.0, 1.0]),
        ]
        neighbourhood = self._neighbourhood()
        context = MOEADContext(
            CyclicSequence(4), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 0
        context.neighbor_type = "NEIGHBOR"
        replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=4
        )

        # Dominates every weight vector's current incumbent.
        offspring = [_solution([0.0, 0.0])]
        result = replacement.replace(list(population), offspring)

        assert any(s.objectives == [0.0, 0.0] for s in result)

    def test_never_replaces_more_than_the_configured_cap(self):
        population = [
            _solution([1.0, 0.0]),
            _solution([0.75, 0.25]),
            _solution([0.25, 0.75]),
            _solution([0.0, 1.0]),
        ]
        neighbourhood = self._neighbourhood()
        context = MOEADContext(
            CyclicSequence(4), neighbourhood_selection_probability=0.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 0
        context.neighbor_type = "POPULATION"
        replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=1
        )

        offspring = [_solution([0.0, 0.0])]
        result = replacement.replace(list(population), offspring)

        replaced_count = sum(1 for s in result if s.objectives == [0.0, 0.0])
        assert replaced_count == 1

    def test_neighbor_scope_scans_in_the_neighborhoods_precomputed_order_unshuffled(self):
        population = [
            _solution([1.0, 0.0]),
            _solution([0.75, 0.25]),
            _solution([0.25, 0.75]),
            _solution([0.0, 1.0]),
        ]
        neighbourhood = self._neighbourhood()
        context = MOEADContext(
            CyclicSequence(4), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(99)
        )
        context.current_subproblem_id = 0
        context.neighbor_type = "NEIGHBOR"

        # A replacement that never actually improves anything -- purely to observe
        # that no exception/reshuffling side effect happens and behavior is
        # rng-independent for the NEIGHBOR scope (the classic algorithm doesn't
        # shuffle it either).
        replacement_a = MOEADReplacement(
            MOEADContext(CyclicSequence(4), 1.0, rng=np.random.default_rng(1)),
            neighbourhood,
            WeightedSum(),
            max_number_of_replaced_solutions=1,
        )
        replacement_a.context.current_subproblem_id = 0
        replacement_a.context.neighbor_type = "NEIGHBOR"

        replacement_b = MOEADReplacement(
            MOEADContext(CyclicSequence(4), 1.0, rng=np.random.default_rng(2)),
            neighbourhood,
            WeightedSum(),
            max_number_of_replaced_solutions=1,
        )
        replacement_b.context.current_subproblem_id = 0
        replacement_b.context.neighbor_type = "NEIGHBOR"

        offspring = [_solution([0.0, 0.0])]
        result_a = replacement_a.replace(list(population), offspring)
        result_b = replacement_b.replace(list(population), offspring)

        # Different rng seeds must not change which solution gets replaced in
        # NEIGHBOR scope, since that scan order is never shuffled.
        assert [s.objectives for s in result_a] == [s.objectives for s in result_b]

    def test_updates_the_aggregation_functions_ideal_point(self):
        population = [_solution([1.0, 1.0]), _solution([0.5, 0.5])]
        neighbourhood = WeightVectorNeighborhood(number_of_weight_vectors=2, neighborhood_size=2)
        context = MOEADContext(
            CyclicSequence(2), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 0
        context.neighbor_type = "NEIGHBOR"
        aggregation_function = Tschebycheff(dimension=2)
        replacement = MOEADReplacement(
            context, neighbourhood, aggregation_function, max_number_of_replaced_solutions=1
        )

        replacement.replace(list(population), [_solution([0.1, 0.1])])

        assert aggregation_function.ideal_point.point == [0.1, 0.1]
