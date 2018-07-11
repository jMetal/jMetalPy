import unittest

from jmetal.problem.singleobjective.unconstrained import OneMax, Sphere


class OneMaxTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = OneMax()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = OneMax()
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(256, problem.number_of_bits)

    def test_should_constructor_create_a_valid_problem_with_512_bits(self) -> None:
        problem = OneMax(512)
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(512, problem.number_of_bits)

    def test_should_create_solution_a_valid_binary_solution(self) -> None:
        problem = OneMax(256)
        solution = problem.create_solution()
        self.assertEqual(256, len(solution.variables[0]))

    def test_should_evaluate_work_properly_if_the_bitset_only_contains_zeroes(self) -> None:
        problem = OneMax(512)
        solution = problem.create_solution()
        solution.variables[0] = [False for _ in range(problem.number_of_bits)]
        problem.evaluate(solution)
        self.assertEqual(0.0, solution.objectives[0])

    def test_should_get_name_return_the_right_name(self):
        problem = OneMax()
        self.assertEqual("OneMax", problem.get_name())


class SphereTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        problem = Sphere(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Sphere()
        self.assertEqual(10, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.12 for _ in range(10)], problem.lower_bound)
        self.assertEqual([5.12 for _ in range(10)], problem.upper_bound)

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
