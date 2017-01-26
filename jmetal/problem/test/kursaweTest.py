import unittest

from jmetal.problem.multiobjective.kursawe import Kursawe

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = Kursawe(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem(self):
        problem = Kursawe(3)
        self.assertEqual(3, solution.get_number_of_variables())
        self.assertEqual(2, solution.get_number_of_objectives())

    def test_should_constructor_create_a_valid_solution_of_floats(self):
        solution = Solution[float](3, 2)

        self.assertEqual(3, solution.get_number_of_variables())
        self.assertEqual(2, solution.get_number_of_objectives())

    def test_should_constructor_create_a_non_none_objective_list(self):
        solution = Solution[float](3.3, 2.42)

        self.assertIsNotNone(solution.objective)

if __name__ == '__main__':
    unittest.main()

