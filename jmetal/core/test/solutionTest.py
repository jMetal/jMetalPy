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
        self.assertEqual(3, solution.get_number_of_objectives())
        self.assertEqual(2, solution.get_number_of_variables())

    def test_should_constructor_create_a_valid_solution_of_floats(self):
        solution = Solution[float](3.3, 2.42)

        self.assertEqual(3.3, solution.get_number_of_objectives())
        self.assertEqual(2.42, solution.get_number_of_variables())

    def test_should_constructor_create_a_non_none_objective_list(self):
        solution = Solution[float](3.3, 2.42)

        self.assertIsNotNone(solution.objective)

if __name__ == '__main__':
    unittest.main()

