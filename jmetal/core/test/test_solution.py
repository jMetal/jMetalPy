import copy
import unittest

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution, Solution


class SolutionTestCase(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = Solution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(0, len(solution.attributes))
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))


class BinarySolutionTestCase(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = BinarySolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False]
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual([True, False], solution.variables[0])
        self.assertEqual([False], solution.variables[1])

    def test_should_get_total_number_of_bits_return_zero_if_the_object_variables_are_not_initialized(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        self.assertEqual(0, solution.get_total_number_of_bits())

    def test_should_get_total_number_of_bits_return_the_right_value(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False, True, False]
        self.assertEqual(5, solution.get_total_number_of_bits())


class FloatSolutionTestCase(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = FloatSolution([], [], 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = FloatSolution([0.0, 0.5], [1.0, 2.0], 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual([0.0, 0.5], solution.lower_bound)
        self.assertEqual([1.0, 2.0], solution.upper_bound)

    def test_should_copy_work_properly(self) -> None:
        solution = FloatSolution([0.0, 3.5], [1.0, 5.0], 3, 2)
        solution.variables = [1.24, 2.66]
        solution.objectives = [0.16, -2.34, 9.25]
        solution.constraints = [-1.2, -0.25]
        solution.attributes["attr"] = "value"

        new_solution = copy.copy(solution)

        self.assertEqual(solution.number_of_variables, new_solution.number_of_variables)
        self.assertEqual(solution.number_of_objectives, new_solution.number_of_objectives)
        self.assertEqual(solution.variables, new_solution.variables)
        self.assertEqual(solution.objectives, new_solution.objectives)
        self.assertEqual(solution.lower_bound, new_solution.lower_bound)
        self.assertEqual(solution.upper_bound, new_solution.upper_bound)
        self.assertEqual(solution.constraints, new_solution.constraints)
        self.assertIs(solution.lower_bound, solution.lower_bound)
        self.assertIs(solution.upper_bound, solution.upper_bound)
        self.assertEqual(solution.attributes, new_solution.attributes)


class IntegerSolutionTestCase(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = IntegerSolution([], [], 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = IntegerSolution([0, 5], [1, 2], 3)

        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual(2, len(solution.variables))
        self.assertEqual(3, len(solution.objectives))
        self.assertEqual([0, 5], solution.lower_bound)
        self.assertEqual([1, 2], solution.upper_bound)

    def test_should_copy_work_properly(self) -> None:
        solution = IntegerSolution([0, 5], [1, 2], 3, 1)
        solution.variables = [1, 2]
        solution.objectives = [0.16, -2.34, 9.25]
        solution.constraints = [-5]
        solution.attributes["attr"] = "value"

        new_solution = copy.copy(solution)

        self.assertEqual(solution.number_of_variables, new_solution.number_of_variables)
        self.assertEqual(solution.number_of_objectives, new_solution.number_of_objectives)
        self.assertEqual(solution.variables, new_solution.variables)
        self.assertEqual(solution.objectives, new_solution.objectives)
        self.assertEqual(solution.lower_bound, new_solution.lower_bound)
        self.assertEqual(solution.upper_bound, new_solution.upper_bound)
        self.assertEqual(solution.constraints, new_solution.constraints)
        self.assertIs(solution.lower_bound, solution.lower_bound)
        self.assertIs(solution.upper_bound, solution.upper_bound)
        self.assertEqual(solution.attributes, new_solution.attributes)


if __name__ == '__main__':
    unittest.main()
