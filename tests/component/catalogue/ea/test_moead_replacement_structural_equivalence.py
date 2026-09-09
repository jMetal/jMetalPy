"""Structural equivalence: MOEADReplacement vs. the classic MOEAD's own replacement.

Not a full-run equivalence test (like test_nsgaii_equivalence.py/
test_smsemoa_equivalence.py): build_moead()/build_moead_de() deliberately draw all
MOEA/D-specific randomness from a single shared rng, never the classic algorithm's mix
of global random and global legacy numpy.random (see MODERNIZATION.md), so two runs
seeded "the same way" can never produce bit-identical fronts. What *can* be verified,
and is verified here, is that MOEADReplacement's decision logic -- given exactly the
same inputs (population, offspring, current subproblem, neighborhood scope) -- makes
exactly the same replace/keep decisions as
jmetal.algorithm.multiobjective.moead.MOEAD.update_current_subproblem_neighborhood(),
the method it generalizes.
"""

import numpy as np

from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.component.catalogue.ea.moead import CyclicSequence, MOEADContext, MOEADReplacement
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.aggregation_function import WeightedSum
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


def _population() -> list[FloatSolution]:
    return [
        _solution([1.0, 0.0]),
        _solution([0.75, 0.25]),
        _solution([0.5, 0.5]),
        _solution([0.25, 0.75]),
        _solution([0.0, 1.0]),
    ]


def _classic_moead_instance(
    neighbourhood: WeightVectorNeighborhood, max_number_of_replaced_solutions: int
) -> MOEAD:
    problem = ZDT1()
    instance = MOEAD(
        problem=problem,
        population_size=5,
        mutation=PolynomialMutation(probability=0.0, distribution_index=20),
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        aggregation_function=WeightedSum(),
        neighbourhood_selection_probability=1.0,
        max_number_of_replaced_solutions=max_number_of_replaced_solutions,
        neighbor_size=2,
        weight_files_path="unused-for-2-objectives",
    )
    instance.neighbourhood = neighbourhood
    return instance


class TestMOEADReplacementMatchesTheClassicUpdateCurrentSubproblemNeighborhood:
    def test_neighbor_scope_produces_an_identical_population(self):
        # WeightVectorNeighborhood's weights/neighbor structure are fully
        # deterministic (analytic for 2 objectives, no randomness involved), so
        # reusing the same instance on both sides is safe and keeps the two subject
        # implementations under test as close to "the same setup" as possible.
        neighbourhood = WeightVectorNeighborhood(number_of_weight_vectors=5, neighborhood_size=2)
        offspring = _solution([0.1, 0.1])

        classic = _classic_moead_instance(neighbourhood, max_number_of_replaced_solutions=1)
        classic.current_subproblem = 2
        classic.neighbor_type = "NEIGHBOR"
        classic_result = classic.update_current_subproblem_neighborhood(offspring, _population())

        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 2
        context.neighbor_type = "NEIGHBOR"
        component_replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=1
        )
        component_result = component_replacement.replace(_population(), [offspring])

        assert [s.objectives for s in component_result] == [s.objectives for s in classic_result]

    def test_neighbor_scope_with_several_replacements_produces_an_identical_population(self):
        neighbourhood = WeightVectorNeighborhood(number_of_weight_vectors=5, neighborhood_size=3)
        # Strictly better than everything in its neighborhood, forcing every
        # candidate the scan visits to be replaced up to the cap.
        offspring = _solution([0.0, 0.0])

        classic = _classic_moead_instance(neighbourhood, max_number_of_replaced_solutions=3)
        classic.current_subproblem = 1
        classic.neighbor_type = "NEIGHBOR"
        classic_result = classic.update_current_subproblem_neighborhood(offspring, _population())

        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 1
        context.neighbor_type = "NEIGHBOR"
        component_replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=3
        )
        component_result = component_replacement.replace(_population(), [offspring])

        assert [s.objectives for s in component_result] == [s.objectives for s in classic_result]

    def test_population_scope_with_no_replacement_cap_produces_an_identical_population(self):
        # With the cap set to the whole population size, every scanned candidate is
        # replaced regardless of the (random, and therefore not forced to match
        # between the two sides here) scan order -- sidestepping the need to
        # reconcile the classic algorithm's global-numpy.random-based permutation
        # with the component version's rng-based one for this comparison.
        neighbourhood = WeightVectorNeighborhood(number_of_weight_vectors=5, neighborhood_size=2)
        # Strictly negative in both objectives: its weighted sum is -(w0 + w1) = -1
        # for any weight vector (weights sum to 1), strictly below every
        # non-negative-objective population member's weighted sum regardless of
        # which weight vector a given index carries -- guaranteeing every scanned
        # candidate loses the comparison, independent of scan order.
        offspring = _solution([-1.0, -1.0])

        classic = _classic_moead_instance(neighbourhood, max_number_of_replaced_solutions=5)
        classic.current_subproblem = 0
        classic.neighbor_type = "POPULATION"
        classic_result = classic.update_current_subproblem_neighborhood(offspring, _population())

        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=0.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 0
        context.neighbor_type = "POPULATION"
        component_replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=5
        )
        component_result = component_replacement.replace(_population(), [offspring])

        assert all(s.objectives == [-1.0, -1.0] for s in component_result)
        assert [s.objectives for s in component_result] == [s.objectives for s in classic_result]

    def test_neither_replaces_more_than_the_configured_cap(self):
        neighbourhood = WeightVectorNeighborhood(number_of_weight_vectors=5, neighborhood_size=4)
        offspring = _solution([0.0, 0.0])

        classic = _classic_moead_instance(neighbourhood, max_number_of_replaced_solutions=2)
        classic.current_subproblem = 3
        classic.neighbor_type = "NEIGHBOR"
        classic_result = classic.update_current_subproblem_neighborhood(offspring, _population())

        context = MOEADContext(
            CyclicSequence(5), neighbourhood_selection_probability=1.0, rng=np.random.default_rng(1)
        )
        context.current_subproblem_id = 3
        context.neighbor_type = "NEIGHBOR"
        component_replacement = MOEADReplacement(
            context, neighbourhood, WeightedSum(), max_number_of_replaced_solutions=2
        )
        component_result = component_replacement.replace(_population(), [offspring])

        classic_replacements = sum(1 for s in classic_result if s.objectives == [0.0, 0.0])
        component_replacements = sum(1 for s in component_result if s.objectives == [0.0, 0.0])
        assert component_replacements == classic_replacements == 2
