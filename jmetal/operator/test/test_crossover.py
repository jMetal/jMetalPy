import unittest
from unittest import mock

from jmetal.core.solution import BinarySolution, PermutationSolution
from jmetal.operator.crossover import NullCrossover, SPXCrossover, CXCrossover, PMXCrossover


class NullCrossoverTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        solution = NullCrossover()
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = NullCrossover()
        self.assertEqual(0, operator.probability)

    def test_should_the_solution_remain_unchanged(self):
        operator = NullCrossover()
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, False], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])


class SinglePointTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            SPXCrossover(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            SPXCrossover(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        solution = SPXCrossover(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = SPXCrossover(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            SPXCrossover(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            SPXCrossover(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = SPXCrossover(0.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, False], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])

    @mock.patch('random_search.randrange')
    def test_should_the_operator_work_if_the_first_bit_is_selected(self, random_call):
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        random_call.return_value = 0
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([False, True, False, False, True, False], offspring[0].variables[0])
        self.assertEqual([True, False, False, True, True, False], offspring[1].variables[0])

    @mock.patch('random_search.randrange')
    def test_should_the_operator_work_if_the_last_bit_is_selected(self, random_call):
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, True]

        random_call.return_value = 5
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, True], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])

    @mock.patch('random_search.randrange')
    def test_should_the_operator_work_if_the_third_bit_is_selected(self, random_call):
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, True]

        random_call.return_value = 3
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, False, True, True], offspring[0].variables[0])
        self.assertEqual([False, True, False, True, True, False], offspring[1].variables[0])

    @mock.patch('random_search.randrange')
    def test_should_the_operator_work_with_a_solution_with_three_binary_variables(self, random_call):
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=3, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution1.variables[1] = [True, False, False, True, False, False]
        solution1.variables[2] = [True, False, True, True, True, True]
        solution2 = BinarySolution(number_of_variables=3, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, True]
        solution2.variables[1] = [True, True, False, False, True, False]
        solution2.variables[2] = [True, True, True, False, False, True]

        random_call.return_value = 8
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, False], offspring[0].variables[0])
        self.assertEqual([True, False, False, False, True, False], offspring[0].variables[1])
        self.assertEqual([True, True, True, False, False, True], offspring[0].variables[2])
        self.assertEqual([False, True, False, False, True, True], offspring[1].variables[0])
        self.assertEqual([True, True, False, True, False, False], offspring[1].variables[1])
        self.assertEqual([True, False, True, True, True, True], offspring[1].variables[2])


class PMXTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            PMXCrossover(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            PMXCrossover(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        solution = PMXCrossover(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = PMXCrossover(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = PMXCrossover(0.0)
        solution1 = PermutationSolution(number_of_variables=2, number_of_objectives=1)
        solution1.variables[0] = [1, 2]
        solution1.variables[1] = [2, 6]

        solution2 = PermutationSolution(number_of_variables=2, number_of_objectives=1)
        solution2.variables[0] = [2, 3]
        solution2.variables[1] = [5, 3]

        offspring = operator.execute([solution1, solution2])

        self.assertEqual([1, 2], offspring[0].variables[0])
        self.assertEqual([2, 6], offspring[0].variables[1])

        self.assertEqual([2, 3], offspring[1].variables[0])
        self.assertEqual([5, 3], offspring[1].variables[1])

    @mock.patch('random_search.randint')
    def test_should_the_operator_work_with_permutation_at_the_middle(self, random_call):
        operator = PMXCrossover(1.0)

        solution1 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution1.variables = [i for i in range(10)]

        solution2 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution2.variables = [i for i in range(10, 20)]

        random_call.side_effect = (2, 4)
        offspring = operator.execute([solution1, solution2])

        self.assertEqual([0, 1, 12, 13, 4, 5, 6, 7, 8, 9], offspring[0].variables)
        self.assertEqual([10, 11, 2, 3, 14, 15, 16, 17, 18, 19], offspring[1].variables)

    @mock.patch('random_search.randint')
    def test_should_the_operator_work_with_permutation_at_the_beginning(self, random_call):
        operator = PMXCrossover(1.0)

        solution1 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution1.variables = [i for i in range(10)]

        solution2 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution2.variables = [i for i in range(10, 20)]

        random_call.side_effect = (0, 5)
        offspring = operator.execute([solution1, solution2])

        self.assertEqual([10, 11, 12, 13, 14, 5, 6, 7, 8, 9], offspring[0].variables)
        self.assertEqual([0, 1, 2, 3, 4, 15, 16, 17, 18, 19], offspring[1].variables)


class CXTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            CXCrossover(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            CXCrossover(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        solution = CXCrossover(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = CXCrossover(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            CXCrossover(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            CXCrossover(-12)

    @mock.patch('random_search.randint')
    def test_should_the_operator_work_with_two_solutions_with_two_variables(self, random_call):
        operator = CXCrossover(1.0)
        solution1 = PermutationSolution(number_of_variables=2, number_of_objectives=1)
        solution1.variables[0] = [1, 2, 3, 4, 7]
        solution1.variables[1] = [2, 6, 4, 5, 3]

        solution2 = PermutationSolution(number_of_variables=2, number_of_objectives=1)
        solution2.variables[0] = [2, 3, 4, 1, 9]
        solution2.variables[1] = [5, 3, 2, 4, 6]

        random_call.return_value = 0
        offspring = operator.execute([solution1, solution2])

        self.assertEqual([1, 2, 3, 4, 9], offspring[0].variables[0])
        self.assertEqual([2, 3, 4, 5, 6], offspring[0].variables[1])

        self.assertEqual([1, 2, 3, 4, 7], offspring[1].variables[0])
        self.assertEqual([2, 6, 4, 5, 3], offspring[1].variables[1])


if __name__ == '__main__':
    unittest.main()
