"""Regression tests for rng support in RE91.

evaluate() draws its own random variables (x7-x10) from a private random.Random()
instance; without an injected rng, that instance is independent of the global
`random` module, so evaluate() was never reproducible via `random.seed()` either
way. Passing rng= at construction makes it reproducible.
"""

import random

import numpy as np

from jmetal.problem.multiobjective.re import RE91


class TestRE91:
    def test_rng_produces_deterministic_objectives_for_a_fixed_seed(self):
        problem_a = RE91(rng=np.random.default_rng(42))
        problem_b = RE91(rng=np.random.default_rng(42))

        random.seed(7)
        solution_a = problem_a.create_solution()
        random.seed(7)
        solution_b = problem_b.create_solution()

        problem_a.evaluate(solution_a)
        problem_b.evaluate(solution_b)

        assert solution_a.objectives == solution_b.objectives

    def test_a_different_seed_gives_different_objectives(self):
        problem_a = RE91(rng=np.random.default_rng(42))
        problem_b = RE91(rng=np.random.default_rng(99))

        random.seed(7)
        solution_a = problem_a.create_solution()
        random.seed(7)
        solution_b = problem_b.create_solution()

        problem_a.evaluate(solution_a)
        problem_b.evaluate(solution_b)

        assert solution_a.objectives != solution_b.objectives

    def test_without_rng_still_produces_a_valid_solution(self):
        problem = RE91()

        solution = problem.create_solution()
        problem.evaluate(solution)

        assert len(solution.objectives) == 9
