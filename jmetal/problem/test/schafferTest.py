import unittest

from jmetal.problem.multiobjectiveproblem.schaffer import Schaffer


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = Schaffer()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Schaffer()
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-100000], problem.lower_bound)
        self.assertEqual([100000], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Schaffer()
        solution = problem.create_solution()

        self.assertEqual(1, solution.number_of_variables)
        self.assertEqual(1, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-100000], problem.lower_bound)
        self.assertEqual([100000], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -100000)
        self.assertTrue(solution.variables[0] <= 100000)

    def test_should_create_solution_return_right_evaluation_values(self):
        problem = Schaffer()

        solution1 = problem.create_solution()
        solution2 = problem.create_solution()
        solution1.variables[0] = 3
        solution2.variables[0] = -2.6

        problem.evaluate(solution1)
        problem.evaluate(solution2)

        self.assertAlmostEqual(solution1.objectives[0], 9);
        self.assertAlmostEqual(solution1.objectives[1], 1);

        self.assertAlmostEqual(solution2.objectives[0], 6.76);
        self.assertAlmostEqual(solution2.objectives[1], 21.16);

    def test_should_get_name_return_the_right_name(self):
        problem = Schaffer()
        self.assertEqual("Schaffer", problem.get_name())

if __name__ == '__main__':
    unittest.main()

