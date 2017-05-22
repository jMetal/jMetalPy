import unittest

from jmetal.core.solution.integerSolution import IntegerSolution
from jmetal.operator.mutation.integerpolynomial import IntegerPolynomial



class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = IntegerPolynomial(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = IntegerPolynomial(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            IntegerPolynomial(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            IntegerPolynomial(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = IntegerPolynomial(0.0)
        solution = IntegerSolution(number_of_variables=2, number_of_objectives=1)
        solution.variables = [1, 2, 3]
        IntegerSolution.lower_bound = [-5, -5, -5]
        IntegerSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])

    def test_should_the_solution_change__if_the_probability_is_one(self):
        operator = IntegerPolynomial(1.0)
        solution = IntegerSolution(number_of_variables=2, number_of_objectives=1)
        solution.variables = [1, 2, 3]
        IntegerSolution.lower_bound = [-5, -5, -5]
        IntegerSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in  mutated_solution.variables])

if __name__ == '__main__':
    unittest.main()
