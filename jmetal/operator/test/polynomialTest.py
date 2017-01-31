import unittest

from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.operator.mutation.polynomial import Polynomial

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = Polynomial(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Polynomial(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            Polynomial(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            Polynomial(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = Polynomial(0.0)
        solution = FloatSolution(number_of_variables=2, number_of_objectives=1)
        solution.variables[0] = [1.0, 2.0, 3.0]
        solution.variables[1] = [-1.0, 4.0, 5.0]
        solution.upper_bound = []

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables[0])
        self.assertEqual([-1.0, 4.0, 5.0], mutated_solution.variables[1])

if __name__ == '__main__':
    unittest.main()
