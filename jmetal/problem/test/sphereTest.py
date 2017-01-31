import unittest

from jmetal.problem.singleobjective.sphere import Sphere

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = Sphere(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Sphere()
        self.assertEqual(10, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.12 for i in range(10)], problem.lower_bound)
        self.assertEqual([5.12 for i in range(10)], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables(self):
        problem = Sphere(5)
        self.assertEqual(5, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.12, -5.12, -5.12, -5.12, -5.12], problem.lower_bound)
        self.assertEqual([5.12, 5.12, 5.12, 5.12, 5.12], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Sphere(3)
        solution = problem.create_solution()

        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(1, solution.number_of_objectives)
        self.assertEqual(1, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.12, -5.12, -5.12], problem.lower_bound)
        self.assertEqual([5.12, 5.12, 5.12], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -5.12)
        self.assertTrue(solution.variables[0] <= 5.12)

    def test_should_evaluate_work_properly(self):
        problem = Sphere(3)
        solution = problem.create_solution()
        problem.evaluate(solution)

        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(1, solution.number_of_objectives)
        self.assertEqual(1, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.12, -5.12, -5.12], problem.lower_bound)
        self.assertEqual([5.12, 5.12, 5.12], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -5.12)
        self.assertTrue(solution.variables[0] <= 5.12)

    def test_should_get_name_return_the_right_name(self):
        problem = Sphere()
        self.assertEqual("Sphere", problem.get_name())

if __name__ == '__main__':
    unittest.main()

