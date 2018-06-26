import unittest

from jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


class ZDT1TestCases(unittest.TestCase):

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


class ZDT2TestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZDT2()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZDT2()
        self.assertEqual(30, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_7_variables(self) -> None:
        problem = ZDT2(7)
        self.assertEqual(7, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(7*[0.0], problem.lower_bound)
        self.assertEqual(7*[1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZDT2()
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
        problem = ZDT2()
        self.assertEqual("ZDT2", problem.get_name())


class ZDT3TestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZDT3()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZDT3()
        self.assertEqual(30, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_7_variables(self) -> None:
        problem = ZDT3(7)
        self.assertEqual(7, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(7*[0.0], problem.lower_bound)
        self.assertEqual(7*[1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZDT3()
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
        problem = ZDT3()
        self.assertEqual("ZDT3", problem.get_name())


class ZDT4TestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZDT4()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZDT4()
        self.assertEqual(10, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(0.0, problem.lower_bound[0])
        self.assertEqual(1.0, problem.upper_bound[0])
        self.assertEqual(9 * [-5.0], problem.lower_bound[1:10])
        self.assertEqual(9 * [5.0], problem.upper_bound[1:10])

    def test_should_constructor_create_a_valid_problem_with_7_variables(self) -> None:
        problem = ZDT4(7)
        self.assertEqual(7, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(0.0, problem.lower_bound[0])
        self.assertEqual(1.0, problem.upper_bound[0])
        self.assertEqual(6 * [-5.0], problem.lower_bound[1:7])
        self.assertEqual(6 * [5.0], problem.upper_bound[1:7])

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZDT4()
        solution = problem.create_solution()
        self.assertEqual(10, solution.number_of_variables)
        self.assertEqual(10, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(0.0, problem.lower_bound[0])
        self.assertEqual(1.0, problem.upper_bound[0])
        self.assertEqual(9 * [-5.0], problem.lower_bound[1:10])
        self.assertEqual(9 * [5.0], problem.upper_bound[1:10])
        self.assertTrue(solution.variables[0] >= -5.0)
        self.assertTrue(solution.variables[0] <= 5.0)
        self.assertTrue(all(value >= -5.0 for value in solution.variables[1:10]))
        self.assertTrue(all(value <= 5.0 for value in solution.variables[1:10]))

    def test_should_get_name_return_the_right_name(self):
        problem = ZDT4()
        self.assertEqual("ZDT4", problem.get_name())


class ZDT6TestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = ZDT6()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = ZDT6()
        self.assertEqual(10, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_7_variables(self) -> None:
        problem = ZDT3(7)
        self.assertEqual(7, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(7*[0.0], problem.lower_bound)
        self.assertEqual(7*[1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = ZDT6()
        solution = problem.create_solution()
        self.assertEqual(10, solution.number_of_variables)
        self.assertEqual(10, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = ZDT6()
        self.assertEqual("ZDT6", problem.get_name())


if __name__ == '__main__':
    unittest.main()
