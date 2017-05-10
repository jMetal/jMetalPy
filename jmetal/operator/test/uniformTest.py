import unittest

from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.operator.mutation.uniform import Uniform


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution1 = Uniform(0.3)
        solution2 = Uniform(0.3, 0.7)
        self.assertIsNotNone(solution1)
        self.assertIsNotNone(solution2)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Uniform(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.perturbation)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            Uniform(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            Uniform(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = Uniform(0.0, 3.0)
        solution = FloatSolution(number_of_variables=3, number_of_objectives=1)
        solution.variables = [1.0, 2.0, 3.0]
        FloatSolution.lower_bound = [-5, -5, -5]
        FloatSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_if_the_probability_is_one(self):
        operator = Uniform(1.0, 3.0)
        solution = FloatSolution(number_of_variables=3, number_of_objectives=1)
        solution.variables = [1.0, 2.0, 3.0]
        FloatSolution.lower_bound = [-5, -5, -5]
        FloatSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_between_max_and_min_value(self):
        operator = Uniform(1.0, 5)
        solution = FloatSolution(number_of_variables=4, number_of_objectives=1)
        solution.variables = [-7.0, 3.0, 12.0, 13.4]
        solution.lower_bound = [-1, 12, -3, -5]
        solution.upper_bound = [1, 17, 3, -2]

        mutated_solution = operator.execute(solution)
        for i in range(solution.number_of_variables):
            self.assertGreaterEqual(mutated_solution.variables[i], solution.lower_bound[i])
            self.assertLessEqual(mutated_solution.variables[i], solution.upper_bound[i])

if __name__ == '__main__':
    unittest.main()
