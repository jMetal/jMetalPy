import unittest

from jmetal.core.solution.binarySolution import BinarySolution

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = BinarySolution(3, 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self):
        solution = BinarySolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution(self):
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)
        solution.variables[0] = [True, False]
        solution.variables[1] = [False]

        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_get_total_number_of_bits_return_zero_if_the_object_variables_are_not_initialized(self):
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)

        self.assertEqual(0, solution.get_total_number_of_bits())

    def test_should_get_total_number_of_bits_return_the_right_value(self):
        solution = BinarySolution(number_of_variables=2, number_of_objectives=3)

        solution.variables[0] = [True, False]
        solution.variables[1] = [False, True, False]

        self.assertEqual(5, solution.get_total_number_of_bits())

if __name__ == '__main__':
    unittest.main()
