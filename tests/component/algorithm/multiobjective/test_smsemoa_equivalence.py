"""Acceptance test: build_smsemoa() must match the classic SMSEMOA.

Same rationale and seeding methodology as test_nsgaii_equivalence.py. Both
implementations reuse the exact same operator classes (SBXCrossover,
PolynomialMutation, RandomSelection, FastNonDominatedRanking,
HypervolumeContributionDensityEstimator) and the exact same Problem.create_solution()
-- so, given identical random state, they must draw the same random numbers in the
same order and therefore produce identical fronts.
"""

import random

import numpy as np
import pytest

from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
from jmetal.component.algorithm.multiobjective.smsemoa import build_smsemoa
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations

_SEED = 42
_POPULATION_SIZE = 20
_MAX_EVALUATIONS = 200


def _operators(problem, seed: int):
    crossover = SBXCrossover(
        probability=1.0, distribution_index=20, rng=np.random.default_rng(seed)
    )
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables(),
        distribution_index=20,
        rng=np.random.default_rng(seed),
    )
    return crossover, mutation


def _run_classic(problem_factory, seed: int) -> list[tuple[float, ...]]:
    problem = problem_factory()
    crossover, mutation = _operators(problem, seed)

    algorithm = SMSEMOA(
        problem=problem,
        population_size=_POPULATION_SIZE,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=_MAX_EVALUATIONS),
    )

    random.seed(seed)
    algorithm.run()

    return sorted(tuple(solution.variables) for solution in algorithm.result())


def _run_component(problem_factory, seed: int) -> list[tuple[float, ...]]:
    problem = problem_factory()
    crossover, mutation = _operators(problem, seed)

    algorithm = build_smsemoa(
        problem,
        population_size=_POPULATION_SIZE,
        crossover=crossover,
        mutation=mutation,
        termination=TerminationByEvaluations(max_evaluations=_MAX_EVALUATIONS),
    )

    random.seed(seed)
    algorithm.run()

    return sorted(tuple(solution.variables) for solution in algorithm.result())


class TestBuildSMSEMOAMatchesTheClassicSMSEMOA:
    @pytest.mark.parametrize("problem_factory", [ZDT1, DTLZ2])
    def test_produces_an_identical_final_population_for_a_fixed_seed(self, problem_factory):
        classic_variables = _run_classic(problem_factory, _SEED)
        component_variables = _run_component(problem_factory, _SEED)

        assert component_variables == classic_variables
