import unittest

from jmetal.core.solution.floatSolution import FloatSolution

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = FloatSolution(3, 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self):
        solution = FloatSolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution(self):
        solution = FloatSolution(3, 2)
        FloatSolution.lower_bound = [1.0 ,2.0, 3.0]
        FloatSolution.upper_bound = [4.0, 5.0, 6.0]
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual([1.0, 2.0, 3.0], solution.lower_bound)
        self.assertEqual([4.0, 5.0, 6.0], solution.upper_bound)
        self.assertEqual(3, len(solution.upper_bound))
        self.assertEqual(3, len(solution.lower_bound))

if __name__ == '__main__':
    unittest.main()
