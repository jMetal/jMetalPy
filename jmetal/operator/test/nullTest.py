import unittest

from jmetal.core.solution import FloatSolution, BinarySolution
from jmetal.operator.mutation import Null


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = Null()
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Null()
        self.assertEqual(0, operator.probability)

    def test_should_the_solution_remain_unchanged_float(self):
        operator = Null()
        solution = FloatSolution(number_of_variables=3, number_of_objectives=1)
        solution.variables = [1.0, 2.0, 3.0]
        solution.lower_bound = [-5, -5, -5]
        solution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_remain_unchanged_binary(self):
        operator = Null()
        solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]

        mutated_solution = operator.execute(solution)
        self.assertEqual([True, True, False, False, True, False], mutated_solution.variables[0])


if __name__ == '__main__':
    unittest.main()
