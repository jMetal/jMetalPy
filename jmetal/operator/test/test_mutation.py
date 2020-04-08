import unittest
from typing import List

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution, CompositeSolution
from jmetal.operator.mutation import BitFlipMutation, UniformMutation, SimpleRandomMutation, PolynomialMutation, \
    IntegerPolynomialMutation, CompositeMutation
from jmetal.util.ckecking import NoneParameterException, EmptyCollectionException, InvalidConditionException


class PolynomialMutationTestMethods(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            PolynomialMutation(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            PolynomialMutation(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        mutation = PolynomialMutation(1.0)
        self.assertIsNotNone(mutation)

    def test_should_constructor_create_a_valid_operator(self):
        operator = PolynomialMutation(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            PolynomialMutation(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            PolynomialMutation(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = PolynomialMutation(0.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 2)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change__if_the_probability_is_one(self):
        operator = PolynomialMutation(1.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 2)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)

        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_execute_work_with_a_solution_subclass_of_float_solution(self):
        class NewFloatSolution(FloatSolution):
            def __init__(self, lower_bound: List[float], upper_bound: List[float], number_of_objectives: int,
                         number_of_constraints: int = 0):
                super(NewFloatSolution, self).__init__(lower_bound, upper_bound, number_of_objectives,
                                                       number_of_constraints)

        operator = PolynomialMutation(1.0)
        solution = NewFloatSolution([-5, -5, -5], [5, 5, 5], 2)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)

        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)


class BitFlipTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            BitFlipMutation(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            BitFlipMutation(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        solution = BitFlipMutation(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = BitFlipMutation(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            BitFlipMutation(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            BitFlipMutation(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = BitFlipMutation(0.0)
        solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]

        mutated_solution = operator.execute(solution)
        self.assertEqual([True, True, False, False, True, False], mutated_solution.variables[0])

    def test_should_the_solution_change_all_the_bits_if_the_probability_is_one(self):
        operator = BitFlipMutation(1.0)
        solution = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]
        solution.variables[1] = [False, True, True, False, False, True]

        mutated_solution = operator.execute(solution)
        self.assertEqual([False, False, True, True, False, True], mutated_solution.variables[0])
        self.assertEqual([True, False, False, True, True, False], mutated_solution.variables[1])


class UniformMutationTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            UniformMutation(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            UniformMutation(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        operator = UniformMutation(0.3)
        operator2 = UniformMutation(0.3, 0.7)
        self.assertIsNotNone(operator)
        self.assertIsNotNone(operator2)

    def test_should_constructor_create_a_valid_operator(self):
        operator = UniformMutation(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.perturbation)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            UniformMutation(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            UniformMutation(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = UniformMutation(0.0, 3.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 1)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_if_the_probability_is_one(self):
        operator = UniformMutation(1.0, 3.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 1)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_between_max_and_min_value(self):
        operator = UniformMutation(1.0, 5)
        solution = FloatSolution([-1, 12, -3, -5], [1, 17, 3, -2], 1)
        solution.variables = [-7.0, 3.0, 12.0, 13.4]

        mutated_solution = operator.execute(solution)
        for i in range(solution.number_of_variables):
            self.assertGreaterEqual(mutated_solution.variables[i], solution.lower_bound[i])
            self.assertLessEqual(mutated_solution.variables[i], solution.upper_bound[i])


class RandomMutationTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            SimpleRandomMutation(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            SimpleRandomMutation(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        operator = SimpleRandomMutation(1.0)
        self.assertIsNotNone(operator)

    def test_should_constructor_create_a_valid_operator(self):
        operator = SimpleRandomMutation(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            SimpleRandomMutation(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            SimpleRandomMutation(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = SimpleRandomMutation(0.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 1)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_if_the_probability_is_one(self):
        operator = SimpleRandomMutation(1.0)
        solution = FloatSolution([-5, -5, -5], [5, 5, 5], 1)
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_between_max_and_min_value(self):
        operator = SimpleRandomMutation(1.0)
        solution = FloatSolution([-1, 12, -3, -5], [1, 17, 3, -2], 1)
        solution.variables = [-7.0, 3.0, 12.0, 13.4]

        mutated_solution = operator.execute(solution)
        for i in range(solution.number_of_variables):
            self.assertGreaterEqual(mutated_solution.variables[i], solution.lower_bound[i])
            self.assertLessEqual(mutated_solution.variables[i], solution.upper_bound[i])


class IntegerPolynomialMutationTestCases(unittest.TestCase):

    def test_should_constructor_raises_an_exception_is_probability_is_negative(self) -> None:
        with self.assertRaises(Exception):
            IntegerPolynomialMutation(-1)

    def test_should_constructor_raises_an_exception_is_probability_is_higher_than_one(self) -> None:
        with self.assertRaises(Exception):
            IntegerPolynomialMutation(1.01)

    def test_should_constructor_create_a_non_null_object(self):
        operator = IntegerPolynomialMutation(1.0)
        self.assertIsNotNone(operator)

    def test_should_constructor_create_a_valid_operator(self):
        operator = IntegerPolynomialMutation(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            IntegerPolynomialMutation(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            IntegerPolynomialMutation(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = IntegerPolynomialMutation(0.0)
        solution = IntegerSolution([-5, -5, -5], [5, 5, 5], 2)
        solution.variables = [1, 2, 3]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])

    def test_should_the_solution_change__if_the_probability_is_one(self):
        operator = IntegerPolynomialMutation(1.0)
        solution = IntegerSolution([-5, -5, -5], [5, 5, 5], 2)
        solution.variables = [1, 2, 3]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])


class CompositeMutationTestCases(unittest.TestCase):
    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_None(self):
        with self.assertRaises(NoneParameterException):
            CompositeMutation(None)

    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_Empty(self):
        with self.assertRaises(EmptyCollectionException):
            CompositeMutation([])

    def test_should_constructor_create_a_valid_operator_when_adding_a_single_mutation_operator(self):
        mutation: Mutation =  PolynomialMutation(0.9, 20.0)

        operator = CompositeMutation([mutation])
        self.assertIsNotNone(operator)
        self.assertEqual(1, len(operator.mutation_operators_list))

    def test_should_constructor_create_a_valid_operator_when_adding_two_mutation_operators(self):
        polynomial_mutation = PolynomialMutation(1.0, 20.0)
        bit_flip_mutation = BitFlipMutation(0.01)

        operator = CompositeMutation([polynomial_mutation, bit_flip_mutation])

        self.assertIsNotNone(operator)
        self.assertEqual(2, len(operator.mutation_operators_list))
        self.assertTrue(issubclass(operator.mutation_operators_list[0].__class__, PolynomialMutation))
        self.assertTrue(issubclass(operator.mutation_operators_list[1].__class__, BitFlipMutation))

    def test_should_execute_raise_and_exception_if_the_types_of_the_solutions_do_not_match_the_operators(self):
        operator = CompositeMutation([PolynomialMutation(1.0, 5.0), PolynomialMutation(0.9, 25.0)])

        float_solution = FloatSolution([2.0], [3.9], 3)
        binary_solution = BinarySolution(1, 3, 0)
        float_solution.variables = [3.0]

        composite_solution = CompositeSolution([float_solution, binary_solution])

        with self.assertRaises(InvalidConditionException):
            operator.execute(composite_solution)


if __name__ == '__main__':
    unittest.main()
