import unittest

from jmetal.core.solution import BinarySolution
from jmetal.operator.mutation import BitFlip

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = BitFlip(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = BitFlip(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            BitFlip(2)

    # comentario
    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            BitFlip(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = BitFlip(0.0)
        solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]

        mutated_solution = operator.execute(solution)
        self.assertEqual([True, True, False, False, True, False], mutated_solution.variables[0])

    def test_should_the_solution_change_all_the_bits_if_the_probability_is_one(self):
        operator = BitFlip(1.0)
        solution = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]
        solution.variables[1] = [False, True, True, False, False, True]

        mutated_solution = operator.execute(solution)
        self.assertEqual([False, False, True, True, False, True], mutated_solution.variables[0])
        self.assertEqual([True, False, False, True, True, False], mutated_solution.variables[1])

if __name__ == '__main__':
    unittest.main()
