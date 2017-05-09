import unittest

from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.core.util.dominancecomparator import dominance_comparator

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_dominance_comparator_raise_an_exception_if_the_first_solution_is_null(self):
        solution = None
        solution2 = FloatSolution(3, 2)
        with self.assertRaises(Exception):
            dominance_comparator(solution, solution2)

    def test_should_dominance_comparator_raise_an_exception_if_the_second_solution_is_null(self):
        solution = FloatSolution(3, 2)
        solution2 = None
        with self.assertRaises(Exception):
            dominance_comparator(solution, solution2)

    def test_should_dominance_comparator_raise_an_exception_if_the_solutions_have_not_the_same_number_of_objectives(self):
        solution = FloatSolution(3, 2)
        solution2 = FloatSolution(3, 5)
        with self.assertRaises(Exception):
            dominance_comparator(solution, solution2)

    def test_should_dominance_comparator_return_zero_if_the_two_solutions_have_one_objective_with_the_same_value(self):
        solution = FloatSolution(3, 1)
        solution2 = FloatSolution(3, 1)

        solution.objectives = [1.0]
        solution2.objectives = [1.0]

        self.assertEqual(0, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_return_one_if_the_two_solutions_have_one_objective_and_the_second_one_is_lower(self):
        solution = FloatSolution(3, 1)
        solution2 = FloatSolution(3, 1)

        solution.objectives = [2.0]
        solution2.objectives = [1.0]

        self.assertEqual(1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_return_minus_one_if_the_two_solutions_have_one_objective_and_the_first_one_is_lower(self):
        solution = FloatSolution(3, 1)
        solution2 = FloatSolution(3, 1)

        solution.objectives = [1.0]
        solution2.objectives = [2.0]

        self.assertEqual(-1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_a(self):
        '''
        Case A: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [2.0, 6.0, 15.0]
        '''
        solution = FloatSolution(3, 3)
        solution2 = FloatSolution(3, 3)

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [2.0, 6.0, 15.0]

        self.assertEqual(-1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_b(self):
        '''
        Case b: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 10.0]
        '''
        solution = FloatSolution(3, 3)
        solution2 = FloatSolution(3, 3)

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 10.0]

        self.assertEqual(-1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_c(self):
        '''
        Case c: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 9.0]
        '''
        solution = FloatSolution(3, 3)
        solution2 = FloatSolution(3, 3)

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 9.0]

        self.assertEqual(1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_d(self):
        '''
        Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-1.0, 5.0, 8.0]
        '''
        solution = FloatSolution(3, 3)
        solution2 = FloatSolution(3, 3)

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-1.0, 5.0, 8.0]

        self.assertEqual(1, dominance_comparator(solution, solution2))

    def test_should_dominance_comparator_work_properly_case_3(self):
        '''
        Case d: solution1 has objectives [-1.0, 5.0, 9.0] and solution2 has [-2.0, 5.0, 10.0]
        '''
        solution = FloatSolution(3, 3)
        solution2 = FloatSolution(3, 3)

        solution.objectives = [-1.0, 5.0, 9.0]
        solution2.objectives = [-2.0, 5.0, 10.0]

        self.assertEqual(0, dominance_comparator(solution, solution2))

if __name__ == '__main__':
    unittest.main()

