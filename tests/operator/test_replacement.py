"""Tests for replacement operators."""

from collections.abc import Callable

import pytest

from jmetal.core.solution import FloatSolution
from jmetal.operator.replacement import (
    RankingAndCrowdingDistanceReplacement,
    RankingAndDensityEstimatorReplacement,
    Replacement,
    SMSEMOAReplacement,
)
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator
from jmetal.util.ranking import FastNonDominatedRanking


class TestReplacement:
    """Tests for the Replacement abstract base class shared by every replacement strategy."""

    @pytest.mark.parametrize(
        "replacement",
        [
            RankingAndDensityEstimatorReplacement(
                FastNonDominatedRanking(), CrowdingDistanceDensityEstimator()
            ),
            RankingAndCrowdingDistanceReplacement(),
            SMSEMOAReplacement(),
        ],
    )
    def test_every_concrete_replacement_is_an_instance_of_replacement(self, replacement):
        """Every existing replacement strategy should honor the shared Replacement contract."""
        assert isinstance(replacement, Replacement)

    def test_replacement_cannot_be_instantiated_directly(self):
        """Replacement is abstract: it only fixes the replace() contract, not an implementation."""
        with pytest.raises(TypeError):
            Replacement()


class TestSMSEMOAReplacement:
    """Tests for the SMS-EMOA replacement operator."""

    @pytest.fixture
    def replacement_2d(self):
        """Create a 2D SMS-EMOA replacement operator."""
        return SMSEMOAReplacement()

    @pytest.fixture
    def replacement_3d(self):
        """Create a 3D SMS-EMOA replacement operator."""
        return SMSEMOAReplacement()

    @pytest.fixture
    def custom_float_solution_factory(self) -> Callable[[list[float]], FloatSolution]:
        """Create a custom float solution factory with specific bounds for testing."""

        def _create_float_solution(objectives: list[float]) -> FloatSolution:
            solution = FloatSolution(
                lower_bound=[0.0] * len(objectives),
                upper_bound=[10.0] * len(objectives),  # Use a larger upper bound for test values
                number_of_objectives=len(objectives),
            )
            solution.objectives = objectives
            solution.variables = [0.5] * len(objectives)  # Initialize variables with default values
            return solution

        return _create_float_solution

    def test_prunes_the_dominated_solution_from_the_last_front_not_the_first(
        self, replacement_2d, custom_float_solution_factory
    ):
        """[4, 4] is dominated by [4, 2] (front 0 = the other four points, front 1 = [4, 4]

        alone) so it must be the one pruned -- regardless of which point in front 0 has
        the smallest hypervolume contribution. This is the behavior the pre-fix
        implementation got wrong: it only ever looked at front 0, so it could remove a
        non-dominated point while leaving the dominated [4, 4] in the result.
        """
        solutions = [
            custom_float_solution_factory(objectives=[5, 1]),
            custom_float_solution_factory(objectives=[1, 5]),
            custom_float_solution_factory(objectives=[4, 2]),
            custom_float_solution_factory(objectives=[4, 4]),
        ]
        dominated = solutions[3]
        offspring = [custom_float_solution_factory(objectives=[5, 1])]

        result = replacement_2d.replace(solutions, offspring)

        assert len(result) == len(solutions) + len(offspring) - 1
        assert dominated not in result
        assert all(s in result for s in solutions[:3] + offspring)

    def test_iteratively_prunes_the_last_front_when_several_solutions_are_excess(
        self, replacement_2d, custom_float_solution_factory
    ):
        """When more than one solution overflows the target size (offspring_population_size

        > 1), every excess solution comes from the single mutually non-dominated front, so
        the iterative removal loop -- not just a single removal -- must run.
        """
        solutions = [
            custom_float_solution_factory(objectives=[1, 5]),
            custom_float_solution_factory(objectives=[2, 4]),
            custom_float_solution_factory(objectives=[3, 3]),
        ]
        offspring = [
            custom_float_solution_factory(objectives=[4, 2]),
            custom_float_solution_factory(objectives=[5, 1]),
        ]

        result = replacement_2d.replace(solutions, offspring)

        assert len(result) == len(solutions)
        assert all(s in solutions + offspring for s in result)

    @pytest.mark.parametrize("n_identical", [2, 3, 5])
    def test_handles_identical_solutions(
        self, replacement_2d, custom_float_solution_factory, n_identical
    ):
        """Test that the operator handles identical solutions correctly."""
        # Given: A set of identical solutions and an identical offspring
        solutions = [custom_float_solution_factory(objectives=[2, 3]) for _ in range(n_identical)]
        offspring = [custom_float_solution_factory(objectives=[1, 1])]

        # When: Applying the replacement
        result = replacement_2d.replace(solutions, offspring)

        # Then: The result should have the correct length
        assert len(result) == len(solutions) + len(offspring) - 1

    def test_works_with_high_dimensional_solutions(
        self, replacement_3d, custom_float_solution_factory
    ):
        """Test that the operator works with 3D solutions."""
        # Given: A set of 3D solutions and an offspring
        solutions = [
            custom_float_solution_factory(objectives=[1, 2, 3]),
            custom_float_solution_factory(objectives=[2, 3, 4]),
            custom_float_solution_factory(objectives=[3, 4, 5]),
        ]
        offspring = [custom_float_solution_factory(objectives=[4, 5, 6])]

        # When: Applying the replacement
        result = replacement_3d.replace(solutions, offspring)

        # Then: The result should have the correct length
        assert len(result) == len(solutions) + len(offspring) - 1

        # And: All solutions should have 3 objectives
        assert all(len(s.objectives) == 3 for s in result)

    def test_preserves_solution_attributes(self, replacement_2d, custom_float_solution_factory):
        """Test that solution attributes are preserved during replacement."""
        # Given: Multiple solutions with custom attributes and one with a specific attribute
        solutions = [
            custom_float_solution_factory(objectives=[2, 3]),
            custom_float_solution_factory(objectives=[1, 4]),
            custom_float_solution_factory(objectives=[3, 2]),
        ]
        solutions[1].attributes["custom_attr"] = "test_value"

        # When: Applying the replacement with one offspring
        offspring = [custom_float_solution_factory(objectives=[1.5, 3.5])]
        result = replacement_2d.replace(solutions, offspring)

        # Then: The result should contain the original solutions plus the offspring minus one
        assert len(result) == len(solutions) + len(offspring) - 1

        # And: If the solution with custom_attr is in the result, its attributes should be preserved
        for solution in result:
            if hasattr(solution, "attributes") and "custom_attr" in solution.attributes:
                assert solution.attributes["custom_attr"] == "test_value"
