import unittest

from jmetal.problem.multiobjective.unconstrained import Kursawe, Fonseca, Schaffer, Viennet2


class KursaweTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Kursawe(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Kursawe()
        self.assertEqual(3, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables(self) -> None:
        problem = Kursawe(5)
        self.assertEqual(5, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.0, -5.0, -5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = Kursawe(3)
        solution = problem.create_solution()
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)
        self.assertTrue(all(variable >= -5.0 for variable in solution.variables))
        self.assertTrue(all(variable <= 5.0 for variable in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = Kursawe()
        self.assertEqual("Kursawe", problem.get_name())


class FonsecaTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        problem = Fonseca()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Fonseca()
        self.assertEqual(3, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(3 * [-4], problem.lower_bound)
        self.assertEqual(3 * [4], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Fonseca()
        solution = problem.create_solution()

        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(3 * [-4], problem.lower_bound)
        self.assertEqual(3 * [4], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -4)
        self.assertTrue(solution.variables[0] <= 4)

    def test_should_create_solution_return_right_evaluation_values(self):
        problem = Fonseca()
        solution1 = problem.create_solution()

        solution1.variables[0] = -1.3
        solution1.variables[1] = 1.5
        solution1.variables[2] = 1.21

        problem.evaluate(solution1)

        self.assertAlmostEqual(solution1.objectives[0], 0.991563628, 4)
        self.assertAlmostEqual(solution1.objectives[1], 0.999663388, 4)

    def test_should_get_name_return_the_right_name(self):
        problem = Fonseca()
        self.assertEqual("Fonseca", problem.get_name())


class SchafferTestCases(unittest.TestCase):

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

        self.assertAlmostEqual(solution1.objectives[0], 9)
        self.assertAlmostEqual(solution1.objectives[1], 1)

        self.assertAlmostEqual(solution2.objectives[0], 6.76)
        self.assertAlmostEqual(solution2.objectives[1], 21.16)

    def test_should_get_name_return_the_right_name(self):
        problem = Schaffer()
        self.assertEqual("Schaffer", problem.get_name())


class Viennet2TestCases(unittest.TestCase):

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

        self.assertAlmostEqual(solution2.objectives[0], 14.0607692307)
        self.assertAlmostEqual(solution2.objectives[1], -11.8818055555)
        self.assertAlmostEqual(solution2.objectives[2], -11.1532369747)

    def test_should_get_name_return_the_right_name(self):
        problem = Viennet2()
        self.assertEqual("Viennet2", problem.get_name())


if __name__ == '__main__':
    unittest.main()
