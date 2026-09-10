import random
import unittest

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.generator import InjectorGenerator, RandomGenerator


class FakeFloatProblem(FloatProblem):
    """Fake class used only for testing purposes."""

    def __init__(self):
        super().__init__()
        self.lower_bound = [-1.0, -2.0]
        self.upper_bound = [1.0, 2.0]

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        return solution

    def name(self) -> str:
        return "Dummy float problem"


class RandomGeneratorTestCase(unittest.TestCase):
    def test_new_forwards_rng_to_create_solution(self):
        # Guards against RandomGenerator silently discarding a caller-supplied
        # rng, which would make population creation unseedable regardless of
        # any rng threaded down from the algorithm.
        problem = FakeFloatProblem()
        generator = RandomGenerator()

        solution_a = generator.new(problem, rng=np.random.default_rng(42))
        solution_b = generator.new(problem, rng=np.random.default_rng(42))

        self.assertEqual(solution_a.variables, solution_b.variables)

    def test_new_without_rng_still_falls_back_to_the_global_random_module(self):
        problem = FakeFloatProblem()
        generator = RandomGenerator()

        random.seed(123)
        solution_a = generator.new(problem)
        random.seed(123)
        solution_b = generator.new(problem)

        self.assertEqual(solution_a.variables, solution_b.variables)


class InjectorGeneratorTestCase(unittest.TestCase):
    def test_new_returns_injected_solutions_before_generating(self):
        problem = FakeFloatProblem()
        injected = problem.create_solution()
        generator = InjectorGenerator([injected])

        result = generator.new(problem, rng=np.random.default_rng(42))

        self.assertEqual(injected.variables, result.variables)

    def test_new_forwards_rng_once_the_injected_population_is_exhausted(self):
        problem = FakeFloatProblem()
        generator = InjectorGenerator([])

        solution_a = generator.new(problem, rng=np.random.default_rng(42))
        solution_b = generator.new(problem, rng=np.random.default_rng(42))

        self.assertEqual(solution_a.variables, solution_b.variables)


if __name__ == "__main__":
    unittest.main()
