import unittest

from jmetal.core.solution.solution import Solution

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = Solution[int](3, 2)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_solution_of_ints(self):
        solution = Solution[int](3, 2)
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)

    def test_should_constructor_create_a_valid_solution_of_floats(self):
        solution = Solution[float](3, 2)

        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(2, solution.number_of_objectives)

    def test_should_constructor_create_a_non_none_objective_list(self):
        solution = Solution[float](3, 2)

        self.assertIsNotNone(solution.objective)

if __name__ == '__main__':
    unittest.main()

