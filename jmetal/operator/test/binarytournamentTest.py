import unittest
from hamcrest import *
from typing import List

from jmetal.core.solution import FloatSolution
from jmetal.operator.selection import BinaryTournament

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        selection = BinaryTournament[FloatSolution]()

        self.assertIsNotNone(selection)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selection = BinaryTournament[FloatSolution]()
        solution_list = None
        with self.assertRaises(Exception):
            selection.execute(solution_list)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selection = BinaryTournament[FloatSolution]()
        solution_list = []
        with self.assertRaises(Exception):
           selection.execute(solution_list)

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self):
        selection = BinaryTournament[FloatSolution]()
        solution = FloatSolution(3,2)
        solution_list = [solution]

        self.assertEqual(solution, selection.execute(solution_list))

    def test_should_execute_work_if_the_solution_list_contains_two_non_dominated_solutions(self):
        selection = BinaryTournament[FloatSolution]()
        solution1 = FloatSolution(2,2)
        solution1.variables = [1.0, 2.0]
        solution2 = FloatSolution(2,2)
        solution2.variables = [0.0, 3.0]

        solution_list = [solution1, solution2]

        assert_that(any_of(solution1 , solution2), selection.execute(solution_list))

    def test_should_execute_work_if_the_solution_list_contains_two_solutions_and_one_them_is_dominated(self):
        selection = BinaryTournament[FloatSolution]()
        solution1 = FloatSolution(2,2)
        solution1.variables = [1.0, 4.0]
        solution2 = FloatSolution(2,2)
        solution2.variables = [0.0, 3.0]

        solution_list = [solution1, solution2]

        assert_that(solution2, selection.execute(solution_list))

if __name__ == '__main__':
    unittest.main()
