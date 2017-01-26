import unittest

from jmetal.problem.singleobjective.onemax import OneMax

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        problem = OneMax()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self):
        problem = OneMax()
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(256, problem.number_of_bits)

    def test_should_constructor_create_a_valid_problem_with_512_bits(self):
        problem = OneMax(512)
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(1, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual(512, problem.number_of_bits)

    def test_should_create_solution_a_valid_binary_solution(self):
        problem = OneMax(256)
        solution = problem.create_solution()

        self.assertEqual(256, len(solution.variables[0]))

    def test_should_evaluate_work_properly_if_the_bitset_only_contains_ones(self):
        problem = OneMax(512)
        solution = problem.create_solution()
        solution.variables[0] = [True for i in range(problem.number_of_bits)]

        problem.evaluate(solution)
        self.assertEqual(512.0, solution.objectives[0])

    def test_should_evaluate_work_properly_if_the_bitset_only_contains_zeroes(self):
        problem = OneMax(512)
        solution = problem.create_solution()
        solution.variables[0] = [False for i in range(problem.number_of_bits)]

        problem.evaluate(solution)
        self.assertEqual(0.0, solution.objectives[0])

if __name__ == '__main__':
    unittest.main()

