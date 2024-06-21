import unittest

from jmetal.problem.multiobjective.unconstrained import (
    Fonseca,
    Kursawe,
    Schaffer,
    Viennet2, OneZeroMax,
)


class KursaweTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Kursawe(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Kursawe()
        self.assertEqual(3, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables(self) -> None:
        problem = Kursawe(5)
        self.assertEqual(5, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual([-5.0, -5.0, -5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = Kursawe(3)
        solution = problem.create_solution()
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)
        self.assertTrue(all(variable >= -5.0 for variable in solution.variables))
        self.assertTrue(all(variable <= 5.0 for variable in solution.variables))

    def test_should_get_name_return_the_right_name(self):
        problem = Kursawe()
        self.assertEqual("Kursawe", problem.name())


class FonsecaTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self):
        problem = Fonseca()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Fonseca()
        self.assertEqual(3, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual(3 * [-4], problem.lower_bound)
        self.assertEqual(3 * [4], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Fonseca()
        solution = problem.create_solution()

        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())

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
        self.assertEqual("Fonseca", problem.name())


class SchafferTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self):
        problem = Schaffer()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Schaffer()
        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual([-1000], problem.lower_bound)
        self.assertEqual([1000], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Schaffer()
        solution = problem.create_solution()

        self.assertEqual(1, len(solution.variables))
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual([-1000], problem.lower_bound)
        self.assertEqual([1000], problem.upper_bound)

        self.assertTrue(solution.variables[0] >= -1000)
        self.assertTrue(solution.variables[0] <= 1000)

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
        self.assertEqual("Schaffer", problem.name())


class Viennet2TestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self):
        problem = Viennet2()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = Viennet2()
        self.assertEqual(2, problem.number_of_variables())
        self.assertEqual(3, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())

        self.assertEqual([-4, -4], problem.lower_bound)
        self.assertEqual([4, 4], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self):
        problem = Viennet2()
        solution = problem.create_solution()

        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints())

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
        self.assertEqual("Viennet2", problem.name())


class OneZeroMaxTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = OneZeroMax()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = OneZeroMax()
        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(256, problem.total_number_of_bits())

    def test_should_constructor_create_a_valid_problem_with_512_bits(self) -> None:
        problem = OneZeroMax(512)
        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual(512, problem.total_number_of_bits())

    def test_should_create_solution_a_valid_binary_solution(self) -> None:
        problem = OneZeroMax(256)
        solution = problem.create_solution()
        self.assertEqual(256, len(solution.variables[0]))

    def test_should_evaluate_work_properly_if_the_bitset_only_contains_zeroes(self) -> None:
        problem = OneZeroMax(512)
        solution = problem.create_solution()
        solution.variables[0] = [False for _ in range(problem.total_number_of_bits())]
        problem.evaluate(solution)
        self.assertEqual(0.0, solution.objectives[0])
        self.assertEqual(-512, solution.objectives[1])

    def test_should_evaluate_work_properly_if_the_bitset_only_contains_ones(self) -> None:
        problem = OneZeroMax(512)
        solution = problem.create_solution()
        solution.variables[0] = [True for _ in range(problem.total_number_of_bits())]
        problem.evaluate(solution)
        self.assertEqual(-512, solution.objectives[0])
        self.assertEqual(0.0, solution.objectives[1])

    def test_should_get_name_return_the_right_name(self):
        problem = OneZeroMax()
        self.assertEqual("OneZeroMax", problem.name())


if __name__ == "__main__":
    unittest.main()
