import unittest

from jmetal.problem.multiobjective.fonseca import Fonseca


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = Fonseca()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Fonseca()
        self.assertEqual(3, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual( 3 * [-4], problem.lower_bound)
        self.assertEqual( 3 * [ 4], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Fonseca()
        solution = problem.create_solution()

        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(3 * [-4], problem.lower_bound)
        self.assertEqual(3 * [ 4], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -4)
        self.assertTrue(solution.variables[0] <= 4)

    def test_should_create_solution_return_right_evaluation_values(self):
        problem = Fonseca()
        solution1 = problem.create_solution()

        solution1.variables[0] = -1.3
        solution1.variables[1] = 1.5
        solution1.variables[2] = 1.21

        problem.evaluate(solution1)

        self.assertAlmostEqual(solution1.objectives[0], 0.991563628, 4);
        self.assertAlmostEqual(solution1.objectives[1], 0.999663388, 4);

    def test_should_get_name_return_the_right_name(self):
        problem = Fonseca()
        self.assertEqual("Fonseca", problem.get_name())

if __name__ == '__main__':
    unittest.main()

