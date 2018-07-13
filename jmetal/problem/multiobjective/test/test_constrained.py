import unittest
from math import pi

from jmetal.problem.multiobjective.constrained import Srinivas, Tanaka


class SrinivasTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Srinivas()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Srinivas()
        self.assertEqual(2, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(2, problem.number_of_constraints)
        self.assertEqual([-20.0, -20.0], problem.lower_bound)
        self.assertEqual([20.0, 20.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = Srinivas()
        solution = problem.create_solution()
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(2, problem.number_of_constraints)
        self.assertTrue(all(variable >= -20.0 for variable in solution.variables))
        self.assertTrue(all(variable <= 20.0 for variable in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = Srinivas()
        self.assertEqual("Srinivas", problem.get_name())


class TanakaTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Tanaka()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Tanaka()
        self.assertEqual(2, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(2, problem.number_of_constraints)
        self.assertEqual([10e-5, 10e-5], problem.lower_bound)
        self.assertEqual([pi, pi], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = Tanaka()
        solution = problem.create_solution()
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(2, problem.number_of_constraints)
        self.assertTrue(all(variable >= 10e-5 for variable in solution.variables))
        self.assertTrue(all(variable <= pi for variable in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = Tanaka()
        self.assertEqual("Tanaka", problem.get_name())


if __name__ == '__main__':
    unittest.main()
