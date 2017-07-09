import unittest

from jmetal.problem.multiobjective.zdt import ZDT1

__author__ = "Antonio J. Nebro"


class ZDT1TestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZDT1()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZDT1()
        self.assertEqual(30, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables(self) -> None:
        problem = ZDT1(5)
        self.assertEqual(5, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(5*[0.0], problem.lower_bound)
        self.assertEqual(5*[1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZDT1()
        solution = problem.create_solution()
        self.assertEqual(30, solution.number_of_variables)
        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = ZDT1()
        self.assertEqual("ZDT1", problem.get_name())


if __name__ == '__main__':
    unittest.main()
