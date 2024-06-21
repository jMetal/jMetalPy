import unittest

from jmetal.problem import DTLZ1, DTLZ2
from jmetal.problem.multiobjective.dtlz import DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from jmetal.problem.multiobjective.zdt import ZDT1


class DTLZ1TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ1()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ1()
        self.assertEqual(7, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(7 * [0.0], problem.lower_bound)
        self.assertEqual(7 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables_and_4_objectives(self) -> None:
        problem = DTLZ1(5, 4)
        self.assertEqual(5, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(5 * [0.0], problem.lower_bound)
        self.assertEqual(5 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ1()
        solution = problem.create_solution()

        self.assertEqual(7, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(7 * [0.0], problem.lower_bound)
        self.assertEqual(7 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = ZDT1()
        self.assertEqual("ZDT1", problem.name())


class DTLZ2TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ2()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ2()
        self.assertEqual(12, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ2(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ2()
        solution = problem.create_solution()
        self.assertEqual(12, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ2()
        self.assertEqual("DTLZ2", problem.name())

class DTLZ3TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ3()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ3()
        self.assertEqual(12, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ3(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ3()
        solution = problem.create_solution()
        self.assertEqual(12, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ3()
        self.assertEqual("DTLZ3", problem.name())


class DTLZ4TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ4()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ4()
        self.assertEqual(12, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ4(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ4()
        solution = problem.create_solution()
        self.assertEqual(12, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ4()
        self.assertEqual("DTLZ4", problem.name())

class DTLZ5TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ5()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ5()
        self.assertEqual(12, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ5(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ5()
        solution = problem.create_solution()
        self.assertEqual(12, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ5()
        self.assertEqual("DTLZ5", problem.name())


class DTLZ6TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ6()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ6()
        self.assertEqual(12, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ6(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ6()
        solution = problem.create_solution()
        self.assertEqual(12, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(12 * [0.0], problem.lower_bound)
        self.assertEqual(12 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ6()
        self.assertEqual("DTLZ6", problem.name())

class DTLZ7TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = DTLZ7()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = DTLZ7()
        self.assertEqual(22, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(22 * [0.0], problem.lower_bound)
        self.assertEqual(22 * [1.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_10_variables_and_4_objectives(self) -> None:
        problem = DTLZ7(10, 4)
        self.assertEqual(10, problem.number_of_variables())
        self.assertEqual(4, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(10 * [0.0], problem.lower_bound)
        self.assertEqual(10 * [1.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = DTLZ7()
        solution = problem.create_solution()
        self.assertEqual(22, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(22 * [0.0], problem.lower_bound)
        self.assertEqual(22 * [1.0], problem.upper_bound)
        self.assertTrue(all(value >= 0.0 for value in solution.variables))
        self.assertTrue(all(value <= 1.0 for value in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = DTLZ7()
        self.assertEqual("DTLZ7", problem.name())


if __name__ == "__main__":
    unittest.main()
