"""Tests for the SolutionsCreation component."""

import numpy as np

from jmetal.component.catalogue.common.solutions_creation import (
    RandomSolutionsCreation,
    SolutionsCreation,
)
from jmetal.problem.singleobjective.unconstrained import Sphere


class TestRandomSolutionsCreation:
    def test_creates_the_requested_number_of_solutions(self):
        problem = Sphere(number_of_variables=5)
        creation = RandomSolutionsCreation(problem, number_of_solutions_to_create=20)

        population = creation.create()

        assert len(population) == 20

    def test_created_solutions_belong_to_the_problem(self):
        problem = Sphere(number_of_variables=5)
        creation = RandomSolutionsCreation(problem, number_of_solutions_to_create=10)

        population = creation.create()

        assert all(len(solution.variables) == 5 for solution in population)

    def test_satisfies_the_solutions_creation_protocol(self):
        problem = Sphere()
        creation = RandomSolutionsCreation(problem, number_of_solutions_to_create=1)

        assert isinstance(creation, SolutionsCreation)

    def test_a_duck_typed_object_satisfies_the_protocol_without_inheriting_from_it(self):
        class FixedSolutionsCreation:
            def create(self):
                return []

        assert isinstance(FixedSolutionsCreation(), SolutionsCreation)

    def test_rng_produces_a_deterministic_population_for_a_fixed_seed(self):
        problem = Sphere(number_of_variables=5)
        creation_a = RandomSolutionsCreation(
            problem, number_of_solutions_to_create=10, rng=np.random.default_rng(42)
        )
        creation_b = RandomSolutionsCreation(
            problem, number_of_solutions_to_create=10, rng=np.random.default_rng(42)
        )

        population_a = creation_a.create()
        population_b = creation_b.create()

        assert [s.variables for s in population_a] == [s.variables for s in population_b]

    def test_without_rng_defaults_to_none(self):
        problem = Sphere(number_of_variables=5)
        creation = RandomSolutionsCreation(problem, number_of_solutions_to_create=1)

        assert creation.rng is None
