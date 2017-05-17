import unittest

from jmetal.problem.multiobjective.viennet2 import Viennet2

class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = Viennet2()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Viennet2()
        self.assertEqual(2, problem.number_of_variables)
        self.assertEqual(3, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-4, -4], problem.lower_bound)
        self.assertEqual([4, 4], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Viennet2()
        solution = problem.create_solution()

        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-4, -4], problem.lower_bound)
        self.assertEqual([4, 4], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -4)
        self.assertTrue(solution.variables[0] <= 4)

    def test_should_create_solution_return_right_evaluation_values(self):
        problem = Viennet2()
        solution2 = problem.create_solution()
        solution2.variables[0] = -2.6
        solution2.variables[1] = 1.5

        problem.evaluate(solution2)

        self.assertAlmostEqual(solution2.objectives[0],  14.0607692307);
        self.assertAlmostEqual(solution2.objectives[1], -11.8818055555);
        self.assertAlmostEqual(solution2.objectives[2], -11.1532369747);

    def test_should_get_name_return_the_right_name(self):
        problem = Viennet2()
        self.assertEqual("Viennet2", problem.get_name())

if __name__ == '__main__':
    unittest.main()

