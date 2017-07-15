import unittest

from jmetal.core.problem import Problem, FloatProblem, IntegerProblem

__author__ = "Antonio J. Nebro"


class ProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = Problem()
        self.assertEqual(0, problem.number_of_variables)
        self.assertEqual(0, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)


class FloatProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = FloatProblem(1, 2, 3, [-1], [1])
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(3, problem.number_of_constraints)
        self.assertEquals([-1], problem.lower_bound)
        self.assertEquals([1], problem.upper_bound)


class IntegerProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = IntegerProblem(1, 2, 3, [-1], [1])
        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(3, problem.number_of_constraints)
        self.assertEquals([-1], problem.lower_bound)
        self.assertEquals([1], problem.upper_bound)
