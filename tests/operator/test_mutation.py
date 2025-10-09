import unittest
from jmetal.operator.mutation import PowerLawMutation

import unittest
import random
from typing import List

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
)
from jmetal.operator.mutation import (
    BitFlipMutation,
    CompositeMutation,
    IntegerPolynomialMutation,
    PolynomialMutation,
    SimpleRandomMutation,
    UniformMutation,
    LevyFlightMutation,
)
from jmetal.util.ckecking import (
    EmptyCollectionException,
    InvalidConditionException,
    NoneParameterException,
)


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
            def __init__(
                    self,
                    lower_bound: List[float],
                    upper_bound: List[float],
                    number_of_objectives: int,
                    number_of_constraints: int = 0,
            ):
                super(NewFloatSolution, self).__init__(
                    lower_bound, upper_bound, number_of_objectives, number_of_constraints
                )

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
        for i in range(len(solution.variables)):
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
        for i in range(len(solution.variables)):
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
        original_variables = solution.variables[:]

        # Try multiple times to account for randomness
        changed = False
        for _ in range(10):  # Try up to 10 times
            solution.variables = original_variables[:]  # Reset to original values
            mutated_solution = operator.execute(solution)
            if mutated_solution.variables != original_variables:
                changed = True
                break

        self.assertTrue(changed, "Solution should change when mutation probability is 1.0")
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])


class CompositeMutationTestCases(unittest.TestCase):
    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_None(self):
        with self.assertRaises(NoneParameterException):
            CompositeMutation(None)

    def test_should_constructor_raise_an_exception_if_the_parameter_list_is_Empty(self):
        with self.assertRaises(EmptyCollectionException):
            CompositeMutation([])

    def test_should_constructor_create_a_valid_operator_when_adding_a_single_mutation_operator(self):
        mutation: Mutation = PolynomialMutation(0.9, 20.0)

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


class LevyFlightMutationTestCases(unittest.TestCase):
    def setUp(self):
        self.lower = [0.0, -5.0, 10.0]
        self.upper = [1.0, 5.0, 20.0]
        self.solution = FloatSolution(self.lower, self.upper, 1)
        self.solution.variables = [0.5, 0.0, 15.0]

    def test_mutation_probability_zero(self):
        mutation = LevyFlightMutation(mutation_probability=0.0)
        mutated = mutation.execute(self.solution)
        self.assertEqual(mutated.variables, [0.5, 0.0, 15.0])

    def test_mutation_probability_one(self):
        mutation = LevyFlightMutation(mutation_probability=1.0)
        mutated = mutation.execute(self.solution)
        # All variables should be mutated (values should change, but remain in bounds)
        for i in range(3):
            self.assertTrue(self.lower[i] <= mutated.variables[i] <= self.upper[i])
            self.assertNotEqual(mutated.variables[i], [0.5, 0.0, 15.0][i])

    def test_beta_parameter(self):
        mutation = LevyFlightMutation(mutation_probability=1.0, beta=1.9)
        mutated = mutation.execute(self.solution)
        for i in range(3):
            self.assertTrue(self.lower[i] <= mutated.variables[i] <= self.upper[i])

    def test_step_size_parameter(self):
        mutation = LevyFlightMutation(mutation_probability=1.0, step_size=0.1)
        mutated = mutation.execute(self.solution)
        for i in range(3):
            self.assertTrue(self.lower[i] <= mutated.variables[i] <= self.upper[i])

    def test_repair_operator(self):
        # Forzar mutación fuera de límites y reparar
        def repair(val, low, up):
            return max(min(val, up), low)

        mutation = LevyFlightMutation(mutation_probability=1.0, step_size=10.0, repair_operator=repair)
        mutated = mutation.execute(self.solution)
        for i in range(3):
            self.assertTrue(self.lower[i] <= mutated.variables[i] <= self.upper[i])

    def test_to_string(self):
        mutation = LevyFlightMutation(mutation_probability=0.5, beta=1.5, step_size=0.01)
        self.assertIn("LevyFlightMutation", str(mutation))


class PowerLawMutationTestCases(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        lower = [0.0, 0.0]
        upper = [10.0, 5.0]
        from jmetal.core.solution import FloatSolution
        self.solution = FloatSolution(lower, upper, number_of_objectives=1)
        self.solution.variables = [5.0, 2.5]

    def test_default_constructor(self):
        mutation = PowerLawMutation()
        self.assertEqual(mutation.probability, 0.01)
        self.assertEqual(mutation.delta, 1.0)

    def test_custom_parameters(self):
        mutation = PowerLawMutation(0.05, 2.0)
        self.assertEqual(mutation.probability, 0.05)
        self.assertEqual(mutation.delta, 2.0)

    def test_invalid_probability(self):
        with self.assertRaises(Exception):
            PowerLawMutation(-0.1, 1.0)
        with self.assertRaises(Exception):
            PowerLawMutation(1.1, 1.0)

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            PowerLawMutation(0.01, 0.0)
        with self.assertRaises(ValueError):
            PowerLawMutation(0.01, -1.0)

    def test_execute_returns_solution(self):
        mutation = PowerLawMutation(0.0, 1.0)
        result = mutation.execute(self.solution)
        self.assertIs(result, self.solution)

    def test_execute_zero_probability(self):
        mutation = PowerLawMutation(0.0, 1.0)
        original = self.solution.variables.copy()
        mutation.execute(self.solution)
        self.assertEqual(self.solution.variables, original)

    def test_execute_always_mutate(self):
        mutation = PowerLawMutation(1.0, 1.0)
        original = self.solution.variables.copy()
        mutation.execute(self.solution)
        self.assertNotEqual(self.solution.variables, original)

    def test_extreme_random_values(self):
        mutation = PowerLawMutation(1.0, 1.0)
        orig_random = random.random
        random.random = lambda: 1e-15
        mutation.execute(self.solution)
        random.random = lambda: 1.0 - 1e-15
        mutation.execute(self.solution)
        random.random = orig_random

    def test_repair_operator(self):
        def repair(val, low, up):
            return low if val < low else up if val > up else val

        mutation = PowerLawMutation(1.0, 1.0, repair_operator=repair)
        mutation.execute(self.solution)

    def test_multiple_variables(self):
        mutation = PowerLawMutation(1.0, 1.0)
        from jmetal.core.solution import FloatSolution
        lower = [0.0, 0.0, 0.0]
        upper = [10.0, 10.0, 10.0]
        sol = FloatSolution(lower, upper, number_of_objectives=1)
        sol.variables = [5.0, 3.0, 7.0]
        mutation.execute(sol)
        self.assertEqual(len(sol.variables), 3)

    def test_str(self):
        mutation = PowerLawMutation(0.05, 2.0)
        self.assertIn("Power Law mutation", mutation.get_name())
        

if __name__ == "__main__":
    unittest.main()
