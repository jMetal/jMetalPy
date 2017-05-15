import unittest

from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.operator.mutation.null import Null


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = Null()
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Null()
        self.assertEqual(0, operator.probability)

    def test_should_the_solution_remain_unchanged(self):
        operator = Null()
        solution = FloatSolution(number_of_variables=3, number_of_objectives=1)
        solution.variables = [1.0, 2.0, 3.0]
        FloatSolution.lower_bound = [-5, -5, -5]
        FloatSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

if __name__ == '__main__':
    unittest.main()
