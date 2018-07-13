import unittest
from unittest import mock

from jmetal.core.solution import BinarySolution
from jmetal.operator.crossover import NullCrossover, SP


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

    def test_should_constructor_create_a_non_null_object(self):
        solution = SP(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = SP(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            SP(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            SP(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = SP(0.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, False], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])

    @mock.patch('random.randrange')
    def test_should_the_operator_work_if_the_first_bit_is_selected(self, random_call):
        operator = SP(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        random_call.return_value = 0
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([False, True, False, False, True, False], offspring[0].variables[0])
        self.assertEqual([True, False, False, True, True, False], offspring[1].variables[0])

    @mock.patch('random.randrange')
    def test_should_the_operator_work_if_the_last_bit_is_selected(self, random_call):
        operator = SP(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, True]

        random_call.return_value = 5
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, True], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])

    @mock.patch('random.randrange')
    def test_should_the_operator_work_if_the_third_bit_is_selected(self, random_call):
        operator = SP(1.0)
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, True]

        random_call.return_value = 3
        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, False, True, True], offspring[0].variables[0])
        self.assertEqual([False, True, False, True, True, False], offspring[1].variables[0])

    @mock.patch('random.randrange')
    def test_should_the_operator_work_with_a_solution_with_three_binary_variables(self, random_call):
        operator = SP(1.0)
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


if __name__ == '__main__':
    unittest.main()
