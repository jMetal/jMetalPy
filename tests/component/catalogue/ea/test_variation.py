"""Tests for the Variation component."""

import pytest

from jmetal.component.catalogue.ea.variation import CrossoverAndMutationVariation, Variation
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Sphere


class TestCrossoverAndMutationVariation:
    def _variation(self, offspring_population_size: int = 10):
        crossover = SBXCrossover(probability=1.0, distribution_index=20)
        mutation = PolynomialMutation(probability=0.1, distribution_index=20)
        return CrossoverAndMutationVariation(offspring_population_size, crossover, mutation)

    def _mating_pool(self, size: int):
        problem = Sphere(number_of_variables=3)
        return [problem.create_solution() for _ in range(size)]

    def test_mating_pool_size_matches_sbx_crossover_arithmetic(self):
        # SBX takes 2 parents and produces 2 children, so mating_pool_size ==
        # offspring_population_size for any even offspring size.
        variation = self._variation(offspring_population_size=10)

        assert variation.mating_pool_size() == 10

    def test_offspring_population_size_is_the_configured_value(self):
        variation = self._variation(offspring_population_size=10)

        assert variation.offspring_population_size() == 10

    def test_variate_produces_the_configured_number_of_offspring(self):
        variation = self._variation(offspring_population_size=10)
        mating_pool = self._mating_pool(variation.mating_pool_size())

        offspring = variation.variate([], mating_pool)

        assert len(offspring) == 10

    def test_variate_rejects_a_mating_pool_with_a_wrong_size(self):
        variation = self._variation(offspring_population_size=10)

        with pytest.raises(ValueError):
            variation.variate([], self._mating_pool(3))

    def test_satisfies_the_variation_protocol(self):
        variation = self._variation()

        assert isinstance(variation, Variation)
