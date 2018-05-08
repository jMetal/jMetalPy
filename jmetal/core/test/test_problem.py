import unittest

from jmetal.core.problem import Problem, FloatProblem, IntegerProblem

__author__ = "Antonio J. Nebro"


class ProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = Problem()
        self.assertEqual(None, problem.number_of_variables)
        self.assertEqual(None, problem.number_of_objectives)
        self.assertEqual(None, problem.number_of_constraints)


class FloatProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = FloatProblem()
        problem.number_of_variables = 1
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1.0]
        problem.upper_bound= [1.0]
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self) -> None:
        problem = FloatProblem()
        problem.number_of_variables = 2
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound= [1.0, 2.0]

        solution = problem.create_solution()
        self.assertNotEqual(None, solution)
        self.assertTrue(-1.0 <= solution.variables[0] <= 1.0)
        self.assertTrue(-2.0 <= solution.variables[1] <= 2.0)


class IntegerProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = IntegerProblem()
        problem.number_of_variables = 1
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1]
        problem.upper_bound= [1]

        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self) -> None:
        problem = FloatProblem()
        problem.number_of_variables = 2
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1, -2]
        problem.upper_bound= [1, 2]

        solution = problem.create_solution()
        self.assertNotEqual(None, solution)
        self.assertTrue(-1 <= solution.variables[0] <= 1)
        self.assertTrue(-2 <= solution.variables[1] <= 2)


if __name__ == '__main__':
    unittest.main()