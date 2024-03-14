import unittest

from jmetal.problem.multiobjective.lz09 import LZ09_F1, LZ09_F2, LZ09_F3, LZ09_F4, LZ09_F5, LZ09_F6, LZ09_F7, LZ09_F8, \
    LZ09_F9


class LZ09F1TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F1()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F1()
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10* [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50(self) -> None:
        problem = LZ09_F1(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F1()
        solution = problem.create_solution()

        self.assertEqual(10, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F1()
        self.assertEqual("LZ09_F1", problem.name())


class LZ09F2TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F2()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F2()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30* [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F2(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F2()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F2()
        self.assertEqual("LZ09_F2", problem.name())


class LZ09F3TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F3()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F3()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30* [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F3(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F3()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F3()
        self.assertEqual("LZ09_F3", problem.name())


class LZ09F4TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F4()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F4()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30* [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F4(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F4()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F4()
        self.assertEqual("LZ09_F4", problem.name())


class LZ09F5TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F5()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F5()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30* [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F5(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F5()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F5()
        self.assertEqual("LZ09_F5", problem.name())


class LZ09F6TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F6()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F6()
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10* [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F6(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F6()
        solution = problem.create_solution()

        self.assertEqual(10, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F6()
        self.assertEqual("LZ09_F6", problem.name())


class LZ09F7TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F7()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F7()
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10* [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F7(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F7()
        solution = problem.create_solution()

        self.assertEqual(10, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F7()
        self.assertEqual("LZ09_F7", problem.name())


class LZ09F8TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F8()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F8()
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10* [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F8(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F8()
        solution = problem.create_solution()

        self.assertEqual(10, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F8()
        self.assertEqual("LZ09_F8", problem.name())


class LZ09F9TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = LZ09_F9()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = LZ09_F9()
        self.assertEqual(30, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30* [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_50_variables(self) -> None:
        problem = LZ09_F9(50)

        self.assertEqual(50, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(50 * [0.0], problem.lower_bound)
        self.assertEqual(50 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = LZ09_F9()
        solution = problem.create_solution()

        self.assertEqual(30, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(30 * [0.0], problem.lower_bound)
        self.assertEqual(30 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = LZ09_F9()
        self.assertEqual("LZ09_F9", problem.name())


if __name__ == "__main__":
    unittest.main()
