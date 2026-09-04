"""Acceptance test for Phase 1: component-based NSGA-II runs are reproducible.

No classic jMetalPy algorithm accepts a seed today, so two runs of the same
configuration are not guaranteed to match (see MODERNIZATION.md's L1 notes). A
component-based algorithm, built from RNG-aware operators plus a reseeded global
`random` state, must produce identical results run to run -- including when
evaluation is parallelized across processes, since MultiprocessEvaluator preserves
input order (it evaluates via Pool.map, not the unordered variant).
"""

import random

import numpy as np

from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.evaluation import SequentialEvaluation
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.evaluator import MultiprocessEvaluator

_SEED = 7
_POPULATION_SIZE = 10
_MAX_EVALUATIONS = 30


def _build_algorithm(seed: int):
    problem = ZDT1()
    crossover = SBXCrossover(
        probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
    )
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(),
        distribution_index=20,
        rng=np.random.default_rng(seed),
    )
    return build_nsgaii(
        problem,
        population_size=_POPULATION_SIZE,
        offspring_population_size=_POPULATION_SIZE,
        crossover=crossover,
        mutation=mutation,
        termination=TerminationByEvaluations(max_evaluations=_MAX_EVALUATIONS),
    )


def _run(seed: int, evaluator=None) -> list[tuple[float, ...]]:
    algorithm = _build_algorithm(seed)
    if evaluator is not None:
        algorithm.evaluation = SequentialEvaluation(algorithm.evaluation.problem, evaluator=evaluator)

    random.seed(seed)
    algorithm.run()

    return sorted(tuple(solution.variables) for solution in algorithm.result())


class TestNSGAIIReproducibility:
    def test_two_runs_with_the_same_seed_produce_the_same_final_population(self):
        first_run = _run(_SEED)
        second_run = _run(_SEED)

        assert first_run == second_run

    def test_a_different_seed_produces_a_different_final_population(self):
        first_run = _run(_SEED)
        other_run = _run(_SEED + 1)

        assert first_run != other_run

    def test_reproducibility_holds_with_a_multiprocess_evaluator(self):
        sequential_run = _run(_SEED)
        multiprocess_run = _run(_SEED, evaluator=MultiprocessEvaluator(processes=2))

        assert multiprocess_run == sequential_run
