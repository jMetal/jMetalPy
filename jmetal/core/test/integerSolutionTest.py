import unittest

from jmetal.core.solution import IntegerSolution


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = IntegerSolution(3, 2)
        self.assertIsNotNone(solution)

    def test_should_default_constructor_create_a_valid_solution(self):
        solution = IntegerSolution(2, 3)
        self.assertEqual(2, solution.number_of_variables)
        self.assertEqual(3, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution(self):
        solution = IntegerSolution(3, 2)
        IntegerSolution.lower_bound = [1 ,2, 3]
        IntegerSolution.upper_bound = [4, 5, 6]
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual([1, 2, 3], solution.lower_bound)
        self.assertEqual([4, 5, 6], solution.upper_bound)
        self.assertEqual(3, len(solution.upper_bound))
        self.assertEqual(3, len(solution.lower_bound))

if __name__ == '__main__':
    unittest.main()
