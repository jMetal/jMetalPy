"""Regression tests for Fase 3's "known remaining gap" fix: several algorithms accept
rng= (Fase 2) but still had internal, algorithm-specific random draws (not going
through any of the four standard operator slots) that ignored it. This covers MOEA/D's
neighbor-type/tour-selection/permutation logic, SMPSO/OMOPSO's velocity-update and
global-best-selection formulas, and NSGA-III's niching().
"""

import numpy as np

from jmetal.algorithm.multiobjective.moead import MOEAD, MOEAD_DRA, Permutation
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import NonUniformMutation, PolynomialMutation, UniformMutation
from jmetal.problem import ZDT1
from jmetal.util.aggregation_function import WeightedSum
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations


class TestPermutation:
    def test_rng_produces_deterministic_sequences_for_a_fixed_seed(self):
        permutation_a = Permutation(10, rng=np.random.default_rng(42))
        permutation_b = Permutation(10, rng=np.random.default_rng(42))

        values_a = [permutation_a.get_next_value() for _ in range(25)]  # spans a reshuffle
        values_b = [permutation_b.get_next_value() for _ in range(25)]

        assert values_a == values_b

    def test_without_rng_still_produces_a_valid_permutation(self):
        permutation = Permutation(10)

        assert sorted(permutation.get_permutation()) == list(range(10))


def _build_moead(seed, cls=MOEAD, **extra):
    kwargs = dict(
        problem=ZDT1(),
        population_size=10,
        mutation=PolynomialMutation(
            probability=0.1, distribution_index=20, rng=np.random.default_rng(seed)
        ),
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, rng=np.random.default_rng(seed)),
        aggregation_function=WeightedSum(),
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        neighbor_size=4,
        weight_files_path="resources/MOEAD_weights",
        termination_criterion=StoppingByEvaluations(max_evaluations=200),
        rng=np.random.default_rng(seed),
    )
    kwargs.update(extra)
    return cls(**kwargs)


class TestMoeadInternalRandomness:
    def test_a_full_run_is_reproducible_for_a_fixed_seed(self):
        algorithm_a = _build_moead(42)
        algorithm_b = _build_moead(42)

        algorithm_a.run()
        algorithm_b.run()

        objectives_a = [tuple(s.objectives) for s in algorithm_a.result()]
        objectives_b = [tuple(s.objectives) for s in algorithm_b.result()]
        assert objectives_a == objectives_b

    def test_moead_dra_tour_selection_is_reproducible_for_a_fixed_seed(self):
        algorithm_a = _build_moead(1, cls=MOEAD_DRA)
        algorithm_b = _build_moead(1, cls=MOEAD_DRA)

        algorithm_a.run()
        algorithm_b.run()

        objectives_a = [tuple(s.objectives) for s in algorithm_a.result()]
        objectives_b = [tuple(s.objectives) for s in algorithm_b.result()]
        assert objectives_a == objectives_b


class TestPsoInternalRandomness:
    def test_smpso_full_run_is_reproducible_for_a_fixed_seed(self):
        def build(seed):
            problem = ZDT1()
            return SMPSO(
                problem=problem,
                swarm_size=10,
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables(),
                    distribution_index=20,
                    rng=np.random.default_rng(seed),
                ),
                leaders=CrowdingDistanceArchive(10),
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        algorithm_a = build(42)
        algorithm_b = build(42)
        algorithm_a.run()
        algorithm_b.run()

        objectives_a = [tuple(s.objectives) for s in algorithm_a.result()]
        objectives_b = [tuple(s.objectives) for s in algorithm_b.result()]
        assert objectives_a == objectives_b

    def test_omopso_full_run_is_reproducible_for_a_fixed_seed(self):
        def build(seed):
            problem = ZDT1()
            return OMOPSO(
                problem=problem,
                swarm_size=10,
                uniform_mutation=UniformMutation(
                    probability=0.1, perturbation=0.5, rng=np.random.default_rng(seed)
                ),
                non_uniform_mutation=NonUniformMutation(
                    probability=0.1,
                    perturbation=0.5,
                    max_iterations=10,
                    rng=np.random.default_rng(seed),
                ),
                leaders=CrowdingDistanceArchive(10),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        algorithm_a = build(42)
        algorithm_b = build(42)
        algorithm_a.run()
        algorithm_b.run()

        objectives_a = [tuple(s.objectives) for s in algorithm_a.result()]
        objectives_b = [tuple(s.objectives) for s in algorithm_b.result()]
        assert objectives_a == objectives_b


class TestNsgaIIINiching:
    def test_full_run_is_reproducible_for_a_fixed_seed(self):
        def build(seed):
            problem = ZDT1()
            return NSGAIII(
                reference_directions=UniformReferenceDirectionFactory(2, n_points=20),
                problem=problem,
                mutation=PolynomialMutation(
                    probability=1.0 / problem.number_of_variables(),
                    distribution_index=20,
                    rng=np.random.default_rng(seed),
                ),
                crossover=SBXCrossover(
                    probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=100),
                rng=np.random.default_rng(seed),
            )

        algorithm_a = build(42)
        algorithm_b = build(42)
        algorithm_a.run()
        algorithm_b.run()

        objectives_a = [tuple(s.objectives) for s in algorithm_a.result()]
        objectives_b = [tuple(s.objectives) for s in algorithm_b.result()]
        assert objectives_a == objectives_b
