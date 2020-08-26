import unittest
from typing import List
from unittest import mock

from jmetal.core.operator import Crossover
from jmetal.core.solution import BinarySolution, PermutationSolution, FloatSolution, CompositeSolution, IntegerSolution
from jmetal.operator.crossover import NullCrossover, SPXCrossover, CXCrossover, PMXCrossover, SBXCrossover, \
    CompositeCrossover, IntegerSBXCrossover
from jmetal.util.ckecking import NoneParameterException, EmptyCollectionException, InvalidConditionException


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

    @mock.patch('random.randrange')
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

    @mock.patch('random.randrange')
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

    @mock.patch('random.randrange')
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

    @mock.patch('random.randrange')
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

    @mock.patch('random.randint')
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

    @mock.patch('random.randint')
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

    @mock.patch('random.randint')
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


class SBXCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_assign_the_correct_probability_value(self):
        crossover_probability = 0.1
        crossover: SBXCrossover = SBXCrossover(crossover_probability, 2.0)

        self.assertEqual(crossover_probability, crossover.probability)

    def test_should_constructor_assign_the_correct_distribution_index_value(self):
        distribution_index = 10.5
        crossover: SBXCrossover = SBXCrossover(0.1, distribution_index)

        self.assertEqual(distribution_index, crossover.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            SBXCrossover(1.5, 2.0)

    def test_should_constructor_raise_an_exception_if_the_probability_is_negative(self):
        with self.assertRaises(Exception):
            SBXCrossover(-0.1, 2.0)

    def test_should_constructor_raise_an_exception_if_the_distribution_index_is_negative(self):
        with self.assertRaises(Exception):
            SBXCrossover(0.1, -2.0)

    def test_should_execute_with_an_invalid_solution_list_size_raise_an_exception(self):
        crossover: SBXCrossover = SBXCrossover(0.1, 20.0)

        solution = FloatSolution([1, 2], [2, 4], 2, 2)
        with self.assertRaises(Exception):
            crossover.execute([solution])

        with self.assertRaises(Exception):
            crossover.execute([solution, solution, solution])

    def test_should_execute_return_the_parents_if_the_crossover_probability_is_zero(self):
        crossover: SBXCrossover = SBXCrossover(0.0, 20.0)

        solution1 = FloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = FloatSolution([1, 2], [2, 4], 2, 2)

        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]

        offspring = crossover.execute([solution1, solution2])

        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)

    def test_should_execute_work_with_a_solution_subclass_of_float_solution(self):
        class NewFloatSolution(FloatSolution):
            def __init__(self, lower_bound: List[float], upper_bound: List[float], number_of_objectives: int,
                         number_of_constraints: int = 0):
                super(NewFloatSolution, self).__init__(lower_bound, upper_bound, number_of_objectives,
                                                       number_of_constraints)

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)

        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]

        crossover: SBXCrossover = SBXCrossover(0.0, 20.0)
        offspring = crossover.execute([solution1, solution2])

        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)


    def test_should_execute_produce_valid_solutions_when_crossing_two_single_variable_solutions(self):
        pass


class CompositeCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_None(self):
        with self.assertRaises(NoneParameterException):
            CompositeCrossover(None)

    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_Empty(self):
        with self.assertRaises(EmptyCollectionException):
            CompositeCrossover([])

    def test_should_constructor_create_a_valid_operator_when_adding_a_single_crossover_operator(self):
        crossover: Crossover = SBXCrossover(0.9, 20.0)

        operator = CompositeCrossover([crossover])
        self.assertIsNotNone(operator)
        self.assertEqual(1, len(operator.crossover_operators_list))

    def test_should_constructor_create_a_valid_operator_when_adding_two_crossover_operators(self):
        sbx_crossover = SBXCrossover(1.0, 20.0)
        single_point_crossover = SPXCrossover(0.01)

        operator = CompositeCrossover([sbx_crossover, single_point_crossover])

        self.assertIsNotNone(operator)
        self.assertEqual(2, len(operator.crossover_operators_list))
        self.assertTrue(issubclass(operator.crossover_operators_list[0].__class__, SBXCrossover))
        self.assertTrue(issubclass(operator.crossover_operators_list[1].__class__, SPXCrossover))

    def test_should_execute_work_properly_with_a_single_crossover_operator(self):
        operator = CompositeCrossover([SBXCrossover(0.9, 20.0)])

        float_solution1 = FloatSolution([2.0], [3.9], 3)
        float_solution1.variables = [3.0]
        float_solution2 = FloatSolution([2.0], [3.9], 3)
        float_solution2.variables = [4.0]

        composite_solution1 = CompositeSolution([float_solution1])
        composite_solution2 = CompositeSolution([float_solution2])

        children = operator.execute([composite_solution1, composite_solution2])

        self.assertIsNotNone(children)
        self.assertEqual(2, len(children))
        self.assertEqual(1, children[0].number_of_variables)
        self.assertEqual(1, children[1].number_of_variables)

    def test_should_execute_work_properly_with_a_two_crossover_operators(self):
        operator = CompositeCrossover([SBXCrossover(0.9, 20.0), IntegerSBXCrossover(0.1, 20.0)])

        float_solution1 = FloatSolution([2.0], [3.9], 3)
        float_solution1.variables = [3.0]
        float_solution2 = FloatSolution([2.0], [3.9], 3)
        float_solution2.variables = [4.0]
        integer_solution1 = IntegerSolution([2], [4], 3)
        integer_solution1.variables = [3.0]
        integer_solution2 = IntegerSolution([2], [7], 3)
        integer_solution2.variables = [4.0]

        composite_solution1 = CompositeSolution([float_solution1, integer_solution1])
        composite_solution2 = CompositeSolution([float_solution2, integer_solution2])

        children = operator.execute([composite_solution1, composite_solution2])

        self.assertIsNotNone(children)
        self.assertEqual(2, len(children))
        self.assertEqual(2, children[0].number_of_variables)
        self.assertEqual(2, children[1].number_of_variables)

    def test_should_execute_raise_and_exception_if_the_types_of_the_solutions_do_not_match_the_operators(self):
        operator = CompositeCrossover([SBXCrossover(1.0, 5.0), SPXCrossover(0.9)])

        float_solution1 = FloatSolution([2.0], [3.9], 3)
        float_solution1.variables = [3.0]
        float_solution2 = FloatSolution([2.0], [3.9], 3)
        float_solution2.variables = [4.0]
        composite_solution1 = CompositeSolution([float_solution1, float_solution2])
        composite_solution2 = CompositeSolution([float_solution1, float_solution2])

        with self.assertRaises(InvalidConditionException):
            operator.execute([composite_solution1, composite_solution2])


if __name__ == '__main__':
    unittest.main()
