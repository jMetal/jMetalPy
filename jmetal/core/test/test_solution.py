import unittest

from jmetal.core.solution import BinarySolution, FloatSolution, Solution

__author__ = "Antonio J. Nebro"


class BinarySolutionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = BinarySolution(3, 2)
        self.assertIsNotNone(solution)

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

    def test_should_get_total_number_of_bits_return_zero_if_the_object_variables_are_not_initialized(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        self.assertEqual(0, solution.get_total_number_of_bits())

    def test_should_get_total_number_of_bits_return_the_right_value(self) -> None:
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False, True, False]
        self.assertEqual(5, solution.get_total_number_of_bits())


class FloatSolutionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = FloatSolution(3, 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self) -> None:
        solution = FloatSolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)
        self.assertEqual([], solution.upper_bound)
        self.assertEqual([], solution.lower_bound)

    def test_should_constructor_create_a_valid_solution(self) -> None:
        solution = FloatSolution(3, 2, [1.0 ,2.0, 3.0],[4.0, 5.0, 6.0])
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual([1.0, 2.0, 3.0], solution.lower_bound)
        self.assertEqual([4.0, 5.0, 6.0], solution.upper_bound)
        self.assertEqual(3, len(solution.upper_bound))
        self.assertEqual(3, len(solution.lower_bound))


class SolutionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_should_constructor_create_a_non_null_object(self) -> None:
        solution = Solution[int](3, 2)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_solution_of_ints(self) -> None:
        solution = Solution[int](3, 2)
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution_of_floats(self) -> None:
        solution = Solution[float](3, 2)
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)

    def test_should_constructor_create_a_non_null_objective_list(self) -> None:
        solution = Solution[float](3, 2)
        self.assertIsNotNone(solution.objectives)


if __name__ == '__main__':
    unittest.main()
