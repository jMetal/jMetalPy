import unittest

from jmetal.core.solution import FloatSolution
from jmetal.operator.selection import RandomSolution


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        selection = RandomSolution[FloatSolution]()

        self.assertIsNotNone(selection)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_none(self):
        selection = RandomSolution[FloatSolution]()
        solution_list = None
        with self.assertRaises(Exception):
            selection.execute(solution_list)

    def test_should_execute_raise_an_exception_if_the_list_of_solutions_is_empty(self):
        selection = RandomSolution[FloatSolution]()
        solution_list = []
        with self.assertRaises(Exception):
           selection.execute(solution_list)

    def test_should_execute_return_the_solution_in_a_list_with_one_solution(self):
        selection = RandomSolution[FloatSolution]()
        solution = FloatSolution(3,2)
        solution_list = [solution]

        self.assertEqual(solution, selection.execute(solution_list))

    def test_should_execute_work_if_the_solution_list_contains_two_non_dominated_solutions(self):
        selection = RandomSolution[FloatSolution]()
        solution1 = FloatSolution(2,2)
        solution1.objectives = [1.0, 2.0]
        solution2 = FloatSolution(2,2)
        solution2.objectives = [0.0, 3.0]

        solution_list = [solution1, solution2]

        self.assertTrue(selection.execute(solution_list) in solution_list)

    def test_should_execute_work_if_the_solution_list_contains_two_solutions_and_one_them_is_dominated(self):
        selection = RandomSolution[FloatSolution]()
        solution1 = FloatSolution(2,2)
        solution1.objectives = [1.0, 4.0]
        solution2 = FloatSolution(2,2)
        solution2.objectives = [0.0, 3.0]

        solution_list = [solution1, solution2]

        self.assertTrue(selection.execute(solution_list) in solution_list)

    def test_should_execute_work_if_the_solution_list_contains_five_solutions_and_one_them_is_dominated(self):
        selection = RandomSolution[FloatSolution]()
        solution1 = FloatSolution(2,2)
        solution1.objectives = [1.0, 4.0]
        solution2 = FloatSolution(2,2)
        solution2.objectives = [0.0, 3.0]
        solution3 = FloatSolution(2,2)
        solution3.objectives = [0.0, 4.0]
        solution4 = FloatSolution(2,2)
        solution4.objectives = [1.0, 3.0]
        solution5 = FloatSolution(2,2)
        solution5.objectives = [0.2, 4.4]

        solution_list = [solution1, solution2, solution3, solution4, solution5]
        self.assertTrue(selection.execute(solution_list) in solution_list)

if __name__ == '__main__':
    unittest.main()
