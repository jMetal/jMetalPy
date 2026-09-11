"""Regression tests for Fase 2 of the RNG-reproducibility plan: a classic algorithm
built with an explicit `rng=` should reproduce its initial population deterministically
and should thread that same generator into any operator that hasn't been seeded
individually, while an algorithm built with no `rng` at all must keep behaving exactly
as before this parameter existed (population creation still consumes the global
`random`/`numpy.random` state, no operator is touched).
"""

import random

import numpy as np

from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, Sphere
from jmetal.util.aggregation_function import WeightedSum


def _zdt1_operators():
    problem = ZDT1()
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(), distribution_index=20
    )
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    return problem, mutation, crossover


class TestRngThreadingIntoTheInitialPopulation:
    def test_nsgaii_with_the_same_seed_produces_the_same_initial_population(self):
        problem, mutation, crossover = _zdt1_operators()

        def build(seed):
            return NSGAII(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
                rng=np.random.default_rng(seed),
            )

        population_a = build(42).create_initial_solutions()
        population_b = build(42).create_initial_solutions()

        assert [s.variables for s in population_a] == [s.variables for s in population_b]

    def test_moead_with_the_same_seed_produces_the_same_initial_population(self):
        def build(seed):
            return MOEAD(
                problem=ZDT1(),
                population_size=5,
                mutation=PolynomialMutation(probability=0.0, distribution_index=20),
                crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
                aggregation_function=WeightedSum(),
                neighbourhood_selection_probability=1.0,
                max_number_of_replaced_solutions=2,
                neighbor_size=2,
                weight_files_path="unused-for-2-objectives",
                rng=np.random.default_rng(42),
            )

        population_a = build(42).create_initial_solutions()
        population_b = build(42).create_initial_solutions()

        assert [s.variables for s in population_a] == [s.variables for s in population_b]

    def test_without_rng_population_creation_still_consumes_the_global_random_module(self):
        # Guards backward compatibility: an algorithm built with no rng= must keep
        # working exactly as before this parameter existed.
        problem, mutation, crossover = _zdt1_operators()

        def build():
            return NSGAII(
                problem=problem,
                population_size=10,
                offspring_population_size=10,
                mutation=mutation,
                crossover=crossover,
            )

        random.seed(123)
        population_a = build().create_initial_solutions()
        random.seed(123)
        population_b = build().create_initial_solutions()

        assert [s.variables for s in population_a] == [s.variables for s in population_b]


class TestRngThreadingIntoOperators:
    def test_an_explicit_rng_is_assigned_to_the_default_selection_operator(self):
        problem, mutation, crossover = _zdt1_operators()
        rng = np.random.default_rng(42)

        algorithm = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=mutation,
            crossover=crossover,
            rng=rng,
        )

        assert algorithm.selection_operator.rng is rng

    def test_an_operator_seeded_explicitly_by_the_caller_is_left_untouched(self):
        problem = Sphere(number_of_variables=3)
        own_rng = np.random.default_rng(1)
        mutation = PolynomialMutation(probability=0.1, distribution_index=20, rng=own_rng)
        crossover = SBXCrossover(probability=0.9, distribution_index=20)

        algorithm = GeneticAlgorithm(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=mutation,
            crossover=crossover,
            rng=np.random.default_rng(42),
        )

        # The mutation operator already had its own rng (eager convention, set at
        # construction) -- the algorithm's rng must not silently override it.
        assert algorithm.mutation_operator.rng is own_rng

    def test_without_rng_no_operator_is_touched(self):
        problem, mutation, crossover = _zdt1_operators()

        algorithm = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            mutation=mutation,
            crossover=crossover,
        )

        assert algorithm.rng is None
        assert algorithm.selection_operator.rng is None


class TestRngThreadingInSingleObjectiveAlgorithms:
    def test_local_search_tie_break_is_deterministic_for_a_fixed_seed(self):
        problem = Sphere(number_of_variables=3)

        from jmetal.util.termination_criterion import StoppingByEvaluations

        def build(seed):
            # A fresh, independently-seeded mutation operator per build: its own
            # rng is set eagerly at construction (not None), so the algorithm's
            # rng is deliberately NOT threaded into it (see the test above) --
            # reusing one mutation instance across both builds would leave its
            # internal state to advance between calls and make this test flaky.
            mutation = PolynomialMutation(
                probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
            )
            return LocalSearch(
                problem=problem,
                mutation=mutation,
                termination_criterion=StoppingByEvaluations(max_evaluations=5),
                rng=np.random.default_rng(seed),
            )

        algorithm_a = build(42)
        algorithm_b = build(42)
        algorithm_a.solutions = algorithm_a.create_initial_solutions()
        algorithm_b.solutions = algorithm_b.create_initial_solutions()
        algorithm_a.solutions = algorithm_a.evaluate(algorithm_a.solutions)
        algorithm_b.solutions = algorithm_b.evaluate(algorithm_b.solutions)

        algorithm_a.step()
        algorithm_b.step()

        assert algorithm_a.solutions[0].variables == algorithm_b.solutions[0].variables
