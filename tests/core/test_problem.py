import random
import unittest

import numpy as np

from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution


class FakeIntegerProblem(IntegerProblem):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self):
        super().__init__()

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        return solution

    def name(self) -> str:
        return "Dummy integer problem"


class FakeFloatProblem(FloatProblem):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self):
        super().__init__()

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        return solution

    def name(self) -> str:
        return "Dummy float problem"


class FloatProblemTestCases(unittest.TestCase):
    def test_should_default_constructor_create_a_valid_problem(self):
        lower_bound = [-1.0]
        upper_bound = [1.0]

        problem = FakeFloatProblem()
        problem.lower_bound = lower_bound
        problem.upper_bound = upper_bound

        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self):
        problem = FakeFloatProblem()

        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound = [1.0, 2.0]

        solution = problem.create_solution()
        self.assertIsNotNone(solution)
        self.assertTrue(-1.0 <= solution.variables[0] <= 1.0)
        self.assertTrue(-2.0 <= solution.variables[1] <= 2.0)

    def test_rng_produces_a_deterministic_solution_for_a_fixed_seed(self):
        problem = FakeFloatProblem()
        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound = [1.0, 2.0]

        solution_a = problem.create_solution(rng=np.random.default_rng(42))
        solution_b = problem.create_solution(rng=np.random.default_rng(42))

        self.assertEqual(solution_a.variables, solution_b.variables)

    def test_without_rng_still_falls_back_to_the_global_random_module(self):
        # Guards the equivalence tests: create_solution() with no rng must keep
        # consuming the seeded global `random` module exactly as before rng= existed,
        # not silently switch to an independent generator.
        problem = FakeFloatProblem()
        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound = [1.0, 2.0]

        random.seed(123)
        solution_a = problem.create_solution()
        random.seed(123)
        solution_b = problem.create_solution()

        self.assertEqual(solution_a.variables, solution_b.variables)


class IntegerProblemTestCases(unittest.TestCase):
    def test_should_default_constructor_create_a_valid_problem(self):
        problem = FakeIntegerProblem()

        problem.lower_bound = [-1]
        problem.upper_bound = [1]

        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self):
        problem = FakeIntegerProblem()

        problem.lower_bound = [-1, -2]
        problem.upper_bound = [1, 2]

        print(problem.number_of_variables())

        solution = problem.create_solution()
        self.assertIsNotNone(solution)
        self.assertTrue(-1 <= solution.variables[0] <= 1)
        self.assertTrue(-2 <= solution.variables[1] <= 2)

    def test_rng_produces_a_deterministic_solution_for_a_fixed_seed(self):
        problem = FakeIntegerProblem()
        problem.lower_bound = [-1, -2]
        problem.upper_bound = [1, 2]

        solution_a = problem.create_solution(rng=np.random.default_rng(42))
        solution_b = problem.create_solution(rng=np.random.default_rng(42))

        self.assertEqual(solution_a.variables, solution_b.variables)


if __name__ == "__main__":
    unittest.main()
