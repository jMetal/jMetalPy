import unittest
from typing import List
from unittest import mock
import numpy as np

from jmetal.core.operator import Crossover
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
)
from jmetal.operator.crossover import (
    ArithmeticCrossover,
    BLXAlphaBetaCrossover,
    BLXAlphaCrossover,
    CompositeCrossover,
    CXCrossover,
    IntegerSBXCrossover,
    NullCrossover,
    PMXCrossover,
    SBXCrossover,
    SPXCrossover,
    UnimodalNormalDistributionCrossover,
)
from jmetal.util.ckecking import Check
from jmetal.util.ckecking import (
    EmptyCollectionException,
    InvalidConditionException,
    NoneParameterException,
)


class NullCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self):
        solution = NullCrossover()
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = NullCrossover()
        self.assertEqual(0, operator.probability)

    def test_should_the_solution_remain_unchanged(self):
        operator = NullCrossover()
        solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution1.bits = np.array([True, False, False, True, True, False])
        solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution2.bits = np.array([False, True, False, False, True, False])

        offspring = operator.execute([solution1, solution2])
        np.testing.assert_array_equal(np.array([True, False, False, True, True, False]), offspring[0].bits)
        np.testing.assert_array_equal(np.array([False, True, False, False, True, False]), offspring[1].bits)


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
        solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution1.bits = np.array([True, False, False, True, True, False])
        solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution2.bits = np.array([False, True, False, False, True, False])

        offspring = operator.execute([solution1, solution2])
        np.testing.assert_array_equal(np.array([True, False, False, True, True, False]), offspring[0].bits)
        np.testing.assert_array_equal(np.array([False, True, False, False, True, False]), offspring[1].bits)

    @mock.patch("numpy.random.default_rng")
    def test_should_the_operator_work_if_the_first_bit_is_selected(self, mock_rng):
        # Mock the numpy random number generator
        mock_rng_instance = mock.MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.random.return_value = 0.1  # Below probability threshold
        mock_rng_instance.integers.return_value = 1  # Crossover point 1
        
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution1.bits = np.array([True, False, False, True, True, False])
        solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution2.bits = np.array([False, True, False, False, True, False])

        offspring = operator.execute([solution1, solution2])
        np.testing.assert_array_equal(np.array([True, True, False, False, True, False]), offspring[0].bits)
        np.testing.assert_array_equal(np.array([False, False, False, True, True, False]), offspring[1].bits)

    @mock.patch("numpy.random.default_rng")
    def test_should_the_operator_work_if_the_last_bit_is_selected(self, mock_rng):
        # Mock the numpy random number generator
        mock_rng_instance = mock.MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.random.return_value = 0.1  # Below probability threshold
        mock_rng_instance.integers.return_value = 5  # Crossover point 5
        
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution1.bits = np.array([True, False, False, True, True, False])
        solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution2.bits = np.array([False, True, False, False, True, True])

        offspring = operator.execute([solution1, solution2])
        np.testing.assert_array_equal(np.array([True, False, False, True, True, True]), offspring[0].bits)
        np.testing.assert_array_equal(np.array([False, True, False, False, True, False]), offspring[1].bits)

    @mock.patch("numpy.random.default_rng")
    def test_should_the_operator_work_if_the_third_bit_is_selected(self, mock_rng):
        # Mock the numpy random number generator
        mock_rng_instance = mock.MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.random.return_value = 0.1  # Below probability threshold
        mock_rng_instance.integers.return_value = 3  # Crossover point 3
        
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution1.bits = np.array([True, False, False, True, True, False])
        solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        solution2.bits = np.array([False, True, False, False, True, True])

        offspring = operator.execute([solution1, solution2])
        np.testing.assert_array_equal(np.array([True, False, False, False, True, True]), offspring[0].bits)
        np.testing.assert_array_equal(np.array([False, True, False, True, True, False]), offspring[1].bits)

    @mock.patch("numpy.random.default_rng")
    def test_should_the_operator_work_with_a_solution_with_three_binary_variables(self, mock_rng):
        # Mock the numpy random number generator
        mock_rng_instance = mock.MagicMock()
        mock_rng.return_value = mock_rng_instance
        mock_rng_instance.random.return_value = 0.1  # Below probability threshold
        mock_rng_instance.integers.return_value = 8  # Crossover point 8
        
        operator = SPXCrossover(1.0)
        solution1 = BinarySolution(number_of_variables=18, number_of_objectives=1)
        solution1.bits = np.array([
            True, False, False, True, True, False,  # var1
            True, False, False, True, False, False,  # var2
            True, False, True, True, True, True      # var3
        ])
        solution2 = BinarySolution(number_of_variables=18, number_of_objectives=1)
        solution2.bits = np.array([
            False, True, False, False, True, True,   # var1
            True, True, False, False, True, False,    # var2
            True, True, True, False, False, True      # var3
        ])

        offspring = operator.execute([solution1, solution2])
        
        # Expected results after crossover at bit 8
        # First 8 bits from parent1, remaining from parent2
        expected_offspring1 = np.array([
            True, False, False, True, True, False,  # var1 (unchanged)
            True, False,                            # first 2 bits of var2 from parent1
            False, False, True, False,              # remaining 4 bits of var2 from parent2
            True, True, True, False, False, True    # var3 from parent2
        ])
        
        # First 8 bits from parent2, remaining from parent1
        expected_offspring2 = np.array([
            False, True, False, False, True, True,  # var1 (unchanged)
            True, True,                             # first 2 bits of var2 from parent2
            False, True, False, False,              # remaining 4 bits of var2 from parent1
            True, False, True, True, True, True     # var3 from parent1
        ])
        
        np.testing.assert_array_equal(expected_offspring1, offspring[0].bits)
        np.testing.assert_array_equal(expected_offspring2, offspring[1].bits)


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
        solution1.variables = [0, 1]

        solution2 = PermutationSolution(number_of_variables=2, number_of_objectives=1)
        solution2.variables = [1, 0]

        offspring = operator.execute([solution1, solution2])

        self.assertEqual([0, 1], offspring[0].variables)
        self.assertEqual([1, 0], offspring[1].variables)

    @mock.patch("random.randint")
    def test_should_the_operator_work_with_permutation_at_the_middle(self, random_call):
        operator = PMXCrossover(1.0)

        solution1 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution1.variables = [i for i in range(10)]

        solution2 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution2.variables = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # Reverse permutation

        random_call.side_effect = (2, 4)
        offspring = operator.execute([solution1, solution2])

        # PMX crossover at positions 2-4 should preserve the middle segment
        # and map the rest according to PMX rules
        self.assertEqual(10, len(offspring[0].variables))
        self.assertEqual(10, len(offspring[1].variables))
        # Verify it's still a valid permutation
        self.assertEqual(sorted(offspring[0].variables), list(range(10)))
        self.assertEqual(sorted(offspring[1].variables), list(range(10)))

    @mock.patch("random.randint")
    def test_should_the_operator_work_with_permutation_at_the_beginning(self, random_call):
        operator = PMXCrossover(1.0)

        solution1 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution1.variables = [i for i in range(10)]

        solution2 = PermutationSolution(number_of_variables=10, number_of_objectives=1)
        solution2.variables = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # Reverse permutation

        random_call.side_effect = (0, 5)
        offspring = operator.execute([solution1, solution2])

        # PMX crossover at positions 0-5 should preserve the first segment
        # and map the rest according to PMX rules
        self.assertEqual(10, len(offspring[0].variables))
        self.assertEqual(10, len(offspring[1].variables))
        # Verify it's still a valid permutation
        self.assertEqual(sorted(offspring[0].variables), list(range(10)))
        self.assertEqual(sorted(offspring[1].variables), list(range(10)))


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

    @mock.patch("random.randint")
    def test_should_the_operator_work_with_two_solutions_with_same_number_of_variables(self, random_call):
        operator = CXCrossover(1.0)
        solution1 = PermutationSolution(number_of_variables=5, number_of_objectives=1)
        solution1.variables = [0, 1, 2, 3, 4]

        solution2 = PermutationSolution(number_of_variables=5, number_of_objectives=1)
        solution2.variables = [1, 2, 3, 0, 4]

        random_call.return_value = 0
        offspring = operator.execute([solution1, solution2])

        # CX crossover should preserve cycles
        self.assertEqual(5, len(offspring[0].variables))
        self.assertEqual(5, len(offspring[1].variables))
        # Verify it's still a valid permutation
        self.assertEqual(sorted(offspring[0].variables), list(range(5)))
        self.assertEqual(sorted(offspring[1].variables), list(range(5)))


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

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)

        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]

        # Test with probability 0.0 (should return parents unchanged)
        crossover = SBXCrossover(0.0, 20.0)
        offspring = crossover.execute([solution1, solution2])

        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)
        
        # Test with probability 1.0 (should perform crossover)
        crossover = SBXCrossover(1.0, 20.0)
        offspring = crossover.execute([solution1, solution2])
        
        self.assertEqual(2, len(offspring))
        self.assertIsInstance(offspring[0], NewFloatSolution)
        self.assertIsInstance(offspring[1], NewFloatSolution)
        
        # Check that the number of objectives and constraints are preserved
        self.assertEqual(solution1.number_of_objectives, offspring[0].number_of_objectives)
        self.assertEqual(solution1.number_of_constraints, offspring[0].number_of_constraints)
        self.assertEqual(solution2.number_of_objectives, offspring[1].number_of_objectives)
        self.assertEqual(solution2.number_of_constraints, offspring[1].number_of_constraints)
        
        # Check that variables are within bounds
        for i in range(len(offspring[0].variables)):
            self.assertGreaterEqual(offspring[0].variables[i], offspring[0].lower_bound[i])
            self.assertLessEqual(offspring[0].variables[i], offspring[0].upper_bound[i])
            self.assertGreaterEqual(offspring[1].variables[i], offspring[1].lower_bound[i])
            self.assertLessEqual(offspring[1].variables[i], offspring[1].upper_bound[i])

    def test_should_execute_produce_valid_solutions_when_crossing_two_single_variable_solutions(self):
        pass

    def test_should_use_correct_bounds_for_each_offspring(self):
        """Test that each offspring uses its own parent's bounds, not mixed bounds."""
        crossover = SBXCrossover(1.0, 20.0)
        
        # Create parents with different bounds
        solution1 = FloatSolution([1.0, 2.0], [3.0, 4.0], 1, 0)  # bounds: [1,3] and [2,4]
        solution2 = FloatSolution([5.0, 6.0], [7.0, 8.0], 1, 0)  # bounds: [5,7] and [6,8]
        
        solution1.variables = [2.0, 3.0]
        solution2.variables = [6.0, 7.0]
        
        offspring = crossover.execute([solution1, solution2])
        
        # Verify that offspring[0] has solution1's bounds
        self.assertEqual(offspring[0].lower_bound, solution1.lower_bound)
        self.assertEqual(offspring[0].upper_bound, solution1.upper_bound)
        
        # Verify that offspring[1] has solution2's bounds
        self.assertEqual(offspring[1].lower_bound, solution2.lower_bound)
        self.assertEqual(offspring[1].upper_bound, solution2.upper_bound)
        
        # Verify that variables are within their respective bounds
        for i in range(len(offspring[0].variables)):
            self.assertGreaterEqual(offspring[0].variables[i], offspring[0].lower_bound[i])
            self.assertLessEqual(offspring[0].variables[i], offspring[0].upper_bound[i])
            self.assertGreaterEqual(offspring[1].variables[i], offspring[1].lower_bound[i])
            self.assertLessEqual(offspring[1].variables[i], offspring[1].upper_bound[i])


class BLXAlphaBetaCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_assign_the_correct_probability_value(self):
        crossover_probability = 0.1
        alpha = 0.3
        beta = 0.5
        crossover: BLXAlphaBetaCrossover = BLXAlphaBetaCrossover(crossover_probability, alpha, beta)

        self.assertEqual(crossover_probability, crossover.probability)
        self.assertEqual(alpha, crossover.alpha)
        self.assertEqual(beta, crossover.beta)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(ValueError):
            BLXAlphaBetaCrossover(1.5, 0.3, 0.5)

    def test_should_constructor_raise_an_exception_if_the_probability_is_negative(self):
        with self.assertRaises(ValueError):
            BLXAlphaBetaCrossover(-0.1, 0.3, 0.5)

    def test_should_constructor_raise_an_exception_if_alpha_is_negative(self):
        with self.assertRaises(ValueError):
            BLXAlphaBetaCrossover(0.1, -0.3, 0.5)

    def test_should_constructor_raise_an_exception_if_beta_is_negative(self):
        with self.assertRaises(ValueError):
            BLXAlphaBetaCrossover(0.1, 0.3, -0.5)

    def test_should_execute_with_an_invalid_solution_list_size_raise_an_exception(self):
        crossover: BLXAlphaBetaCrossover = BLXAlphaBetaCrossover(0.1, 0.3, 0.5)

        solution = FloatSolution([1, 2], [2, 4], 2, 2)
        with self.assertRaises(Exception):
            crossover.execute([solution])

        with self.assertRaises(Exception):
            crossover.execute([solution, solution, solution])

    def test_should_execute_return_the_parents_if_the_crossover_probability_is_zero(self):
        crossover: BLXAlphaBetaCrossover = BLXAlphaBetaCrossover(0.0, 0.3, 0.5)

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

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)

        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]

        crossover: BLXAlphaBetaCrossover = BLXAlphaBetaCrossover(0.0, 0.3, 0.5)
        offspring = crossover.execute([solution1, solution2])

        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)

    def test_should_use_correct_bounds_for_each_offspring(self):
        """Test that each offspring uses its own parent's bounds, not mixed bounds."""
        crossover = BLXAlphaBetaCrossover(1.0, 0.5, 0.3)
        
        # Create parents with different bounds
        solution1 = FloatSolution([1.0, 2.0], [3.0, 4.0], 1, 0)  # bounds: [1,3] and [2,4]
        solution2 = FloatSolution([5.0, 6.0], [7.0, 8.0], 1, 0)  # bounds: [5,7] and [6,8]
        
        solution1.variables = [2.0, 3.0]
        solution2.variables = [6.0, 7.0]
        
        offspring = crossover.execute([solution1, solution2])
        
        # Verify that offspring[0] has solution1's bounds
        self.assertEqual(offspring[0].lower_bound, solution1.lower_bound)
        self.assertEqual(offspring[0].upper_bound, solution1.upper_bound)
        
        # Verify that offspring[1] has solution2's bounds
        self.assertEqual(offspring[1].lower_bound, solution2.lower_bound)
        self.assertEqual(offspring[1].upper_bound, solution2.upper_bound)

    @mock.patch('random.random')
    @mock.patch('random.uniform')
    def test_should_execute_produce_valid_solutions_within_expanded_range(self, uniform_mock, random_mock):
        # Mock random.random() to ensure crossover happens
        random_mock.return_value = 0.05  # Well below probability threshold of 0.9
        
        # Mock random.uniform to return specific values for testing
        uniform_mock.side_effect = [2.5, 3.0, 4.0, 4.5]
        
        crossover: BLXAlphaBetaCrossover = BLXAlphaBetaCrossover(0.9, 0.5, 0.3)
        
        # Create two parent solutions with known values
        solution1 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution2 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        
        # Set parent values that will create known ranges
        solution1.variables = [2.0, 4.0]  # First variable: 2.0, Second: 4.0
        solution2.variables = [3.0, 5.0]  # First variable: 3.0, Second: 5.0
        
        offspring = crossover.execute([solution1, solution2])
        
        self.assertEqual(2, len(offspring))
        
        # Check that the values are within the expected expanded ranges
        # First variable should be in [1.5, 3.3]
        self.assertGreaterEqual(offspring[0].variables[0], 1.5)
        self.assertLessEqual(offspring[0].variables[0], 3.3)
        self.assertGreaterEqual(offspring[1].variables[0], 1.5)
        self.assertLessEqual(offspring[1].variables[0], 3.3)
        
        # Second variable should be in [3.5, 5.0]
        self.assertGreaterEqual(offspring[0].variables[1], 3.5)
        self.assertLessEqual(offspring[0].variables[1], 5.0)
        self.assertGreaterEqual(offspring[1].variables[1], 3.5)
        self.assertLessEqual(offspring[1].variables[1], 5.0)
        
        # Verify the mock was called with the expected arguments
        # We expect 4 calls to uniform: 2 variables * 2 offspring
        self.assertEqual(uniform_mock.call_count, 4)


class BLXAlphaCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_assign_the_correct_probability_value(self):
        crossover_probability = 0.1
        alpha = 0.5
        crossover: BLXAlphaCrossover = BLXAlphaCrossover(crossover_probability, alpha)

        self.assertEqual(crossover_probability, crossover.probability)
        self.assertEqual(alpha, crossover.alpha)

    def test_should_constructor_assign_the_correct_alpha_value(self):
        alpha = 0.3
        crossover: BLXAlphaCrossover = BLXAlphaCrossover(0.1, alpha)

        self.assertEqual(alpha, crossover.alpha)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(ValueError):
            BLXAlphaCrossover(1.5, 0.5)

    def test_should_constructor_raise_an_exception_if_the_probability_is_negative(self):
        with self.assertRaises(ValueError):
            BLXAlphaCrossover(-0.1, 0.5)

    def test_should_constructor_raise_an_exception_if_alpha_is_negative(self):
        with self.assertRaises(ValueError):
            BLXAlphaCrossover(0.1, -0.5)

    def test_should_execute_with_an_invalid_solution_list_size_raise_an_exception(self):
        crossover: BLXAlphaCrossover = BLXAlphaCrossover(0.1, 0.5)

        solution = FloatSolution([1, 2], [2, 4], 2, 2)
        with self.assertRaises(Exception):
            crossover.execute([solution])

        with self.assertRaises(Exception):
            crossover.execute([solution, solution, solution])

    def test_should_execute_return_the_parents_if_the_crossover_probability_is_zero(self):
        crossover: BLXAlphaCrossover = BLXAlphaCrossover(0.0, 0.5)

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

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)

        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]

        crossover: BLXAlphaCrossover = BLXAlphaCrossover(0.0, 0.5)
        offspring = crossover.execute([solution1, solution2])

        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)

    def test_should_use_correct_bounds_for_each_offspring(self):
        """Test that each offspring uses its own parent's bounds, not mixed bounds."""
        crossover = BLXAlphaCrossover(1.0, 0.5)
        
        # Create parents with different bounds
        solution1 = FloatSolution([1.0, 2.0], [3.0, 4.0], 1, 0)  # bounds: [1,3] and [2,4]
        solution2 = FloatSolution([5.0, 6.0], [7.0, 8.0], 1, 0)  # bounds: [5,7] and [6,8]
        
        solution1.variables = [2.0, 3.0]
        solution2.variables = [6.0, 7.0]
        
        offspring = crossover.execute([solution1, solution2])
        
        # Verify that offspring[0] has solution1's bounds
        self.assertEqual(offspring[0].lower_bound, solution1.lower_bound)
        self.assertEqual(offspring[0].upper_bound, solution1.upper_bound)
        
        # Verify that offspring[1] has solution2's bounds
        self.assertEqual(offspring[1].lower_bound, solution2.lower_bound)
        self.assertEqual(offspring[1].upper_bound, solution2.upper_bound)

    @mock.patch('random.random')
    def test_should_execute_produce_valid_solutions_within_expanded_range(self, random_mock):
        # Mock random.random() to return 0.5 (ensuring crossover happens)
        random_mock.return_value = 0.05  # Well below probability threshold of 0.9
        
        crossover: BLXAlphaCrossover = BLXAlphaCrossover(0.9, 0.5)  # 90% probability, alpha=0.5
        
        # Create two parent solutions with known values
        solution1 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution2 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        
        # Set parent values that will create a known range when alpha=0.5
        # For variable 1: min=2, max=3, range=1, expanded range = [1.5, 3.5]
        # For variable 2: min=4, max=5, range=1, expanded range = [3.5, 5.5] (clamped to [3.5, 5.0])
        solution1.variables = [2.0, 4.0]
        solution2.variables = [3.0, 5.0]
        
        # Mock random.uniform to return the lower bound of the expanded range for the first variable
        # and the upper bound for the second variable
        with mock.patch('random.uniform', side_effect=[1.5, 1.5, 5.0, 5.0]):
            offspring = crossover.execute([solution1, solution2])
            
            self.assertEqual(2, len(offspring))
            
            # Check that the values are within the expected expanded ranges
            # First variable should be in [1.5, 3.5]
            self.assertGreaterEqual(offspring[0].variables[0], 1.5)
            self.assertLessEqual(offspring[0].variables[0], 3.5)
            self.assertGreaterEqual(offspring[1].variables[0], 1.5)
            self.assertLessEqual(offspring[1].variables[0], 3.5)
            
            # Second variable should be in [3.5, 5.0] (upper bound clamped)
            self.assertGreaterEqual(offspring[0].variables[1], 3.5)
            self.assertLessEqual(offspring[0].variables[1], 5.0)
            self.assertGreaterEqual(offspring[1].variables[1], 3.5)
            self.assertLessEqual(offspring[1].variables[1], 5.0)


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
        self.assertEqual(1, len(children[0].variables))
        self.assertEqual(1, len(children[1].variables))

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
        self.assertEqual(2, len(children[0].variables))
        self.assertEqual(2, len(children[1].variables))

    def test_should_execute_raise_and_exception_if_the_types_of_the_solutions_do_not_match_the_operators(self):
        operator = CompositeCrossover([SBXCrossover(1.0, 5.0), SPXCrossover(0.9)])

        float_solution1 = FloatSolution([2.0], [3.9], 1)  # Changed to 1 objective
        float_solution1.variables = [3.0]
        float_solution2 = FloatSolution([2.0], [3.9], 1)  # Changed to 1 objective
        float_solution2.variables = [4.0]
        binary_solution1 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        binary_solution1.bits = np.array([True, False, False, True, True, False])
        binary_solution2 = BinarySolution(number_of_variables=6, number_of_objectives=1)
        binary_solution2.bits = np.array([False, True, False, False, True, False])
        
        composite_solution1 = CompositeSolution([float_solution1, binary_solution1])
        composite_solution2 = CompositeSolution([float_solution2, binary_solution2])

        # This should work because we have matching types: FloatSolution for SBX, BinarySolution for SPX
        offspring = operator.execute([composite_solution1, composite_solution2])
        
        # Verify that the crossover worked
        self.assertEqual(2, len(offspring))
        self.assertEqual(2, len(offspring[0].variables))
        self.assertEqual(2, len(offspring[1].variables))



class ArithmeticCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_assign_the_correct_probability_value(self):
        crossover_probability = 0.1
        crossover: ArithmeticCrossover = ArithmeticCrossover(crossover_probability)
        self.assertEqual(crossover_probability, crossover.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(ValueError):
            ArithmeticCrossover(1.5)

    def test_should_constructor_raise_an_exception_if_the_probability_is_negative(self):
        with self.assertRaises(ValueError):
            ArithmeticCrossover(-0.1)

    def test_should_execute_with_an_invalid_solution_list_size_raise_an_exception(self):
        crossover: ArithmeticCrossover = ArithmeticCrossover(0.1)
        solution = FloatSolution([1, 2], [2, 4], 2, 2)
        with self.assertRaises(Exception):
            crossover.execute([solution])
        with self.assertRaises(Exception):
            crossover.execute([solution, solution, solution])

    def test_should_execute_return_the_parents_if_the_crossover_probability_is_zero(self):
        crossover: ArithmeticCrossover = ArithmeticCrossover(0.0)
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

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]
        crossover: ArithmeticCrossover = ArithmeticCrossover(0.0)
        offspring = crossover.execute([solution1, solution2])
        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)

    def test_should_use_correct_bounds_for_each_offspring(self):
        """Test that each offspring uses its own parent's bounds, not mixed bounds."""
        crossover = ArithmeticCrossover(1.0)
        
        # Create parents with different bounds
        solution1 = FloatSolution([1.0, 2.0], [3.0, 4.0], 1, 0)  # bounds: [1,3] and [2,4]
        solution2 = FloatSolution([5.0, 6.0], [7.0, 8.0], 1, 0)  # bounds: [5,7] and [6,8]
        
        solution1.variables = [2.0, 3.0]
        solution2.variables = [6.0, 7.0]
        
        offspring = crossover.execute([solution1, solution2])
        
        # Verify that offspring[0] has solution1's bounds
        self.assertEqual(offspring[0].lower_bound, solution1.lower_bound)
        self.assertEqual(offspring[0].upper_bound, solution1.upper_bound)
        
        # Verify that offspring[1] has solution2's bounds
        self.assertEqual(offspring[1].lower_bound, solution2.lower_bound)
        self.assertEqual(offspring[1].upper_bound, solution2.upper_bound)

    @mock.patch('jmetal.operator.crossover.random.random')
    def test_should_execute_perform_arithmetic_crossover(self, random_mock):
        # Mock random.random() to return 0.1 (for probability check) and 0.5 for alpha
        random_mock.side_effect = [0.1, 0.5]  # 0.1 < 0.9, so crossover happens, alpha=0.5 for all variables
        
        crossover: ArithmeticCrossover = ArithmeticCrossover(0.9)
        
        # Create two parent solutions with known values
        solution1 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution2 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        
        # Set parent values
        solution1.variables = [2.0, 4.0]
        solution2.variables = [3.0, 5.0]
        
        # Expected calculations with alpha=0.5:
        # For first variable: 0.5*2.0 + 0.5*3.0 = 2.5 and 0.5*2.0 + 0.5*3.0 = 2.5
        # For second variable: 0.5*4.0 + 0.5*5.0 = 4.5 and 0.5*4.0 + 0.5*5.0 = 4.5
        # The test expects the same alpha to be used for both variables
        
        offspring = crossover.execute([solution1, solution2])
        
        self.assertEqual(2, len(offspring))
        
        # Check that the values are the arithmetic mean of the parents
        self.assertEqual(offspring[0].variables[0], 2.5)  # (2.0 + 3.0) / 2
        self.assertEqual(offspring[0].variables[1], 4.5)  # (4.0 + 5.0) / 2
        self.assertEqual(offspring[1].variables[0], 2.5)  # (2.0 + 3.0) / 2
        self.assertEqual(offspring[1].variables[1], 4.5)  # (4.0 + 5.0) / 2
        
        # Verify random.random() was called 2 times (1 for probability, 1 for alpha)
        self.assertEqual(random_mock.call_count, 2)
        
        # Reset the mock for the next test
        random_mock.reset_mock()

class UnimodalNormalDistributionCrossoverTestCases(unittest.TestCase):
    def test_should_constructor_assign_the_correct_probability_value(self):
        crossover_probability = 0.1
        zeta = 0.5
        eta = 0.35
        crossover = UnimodalNormalDistributionCrossover(crossover_probability, zeta, eta)
        self.assertEqual(crossover_probability, crossover.probability)
        self.assertEqual(zeta, crossover.zeta)
        self.assertEqual(eta, crossover.eta)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(ValueError):
            UnimodalNormalDistributionCrossover(1.5, 0.5, 0.35)

    def test_should_constructor_raise_an_exception_if_the_probability_is_negative(self):
        with self.assertRaises(ValueError):
            UnimodalNormalDistributionCrossover(-0.1, 0.5, 0.35)

    def test_should_constructor_raise_an_exception_if_zeta_is_negative(self):
        with self.assertRaises(ValueError):
            UnimodalNormalDistributionCrossover(0.9, -0.1, 0.35)

    def test_should_constructor_raise_an_exception_if_eta_is_negative(self):
        with self.assertRaises(ValueError):
            UnimodalNormalDistributionCrossover(0.9, 0.5, -0.1)

    def test_should_execute_with_an_invalid_solution_list_size_raise_an_exception(self):
        crossover = UnimodalNormalDistributionCrossover(0.1, 0.5, 0.35)
        solution = FloatSolution([1, 2], [2, 4], 2, 2)
        
        # Test with too few parents
        with self.assertRaises(Exception):
            crossover.execute([solution])
        with self.assertRaises(Exception):
            crossover.execute([solution, solution])
            
        # Test with too many parents (should still work, just use first three)
        try:
            crossover.execute([solution, solution, solution, solution])
        except Exception as e:
            self.fail(f"Should accept more than 3 parents but got: {e}")

    def test_should_execute_return_the_parents_if_the_crossover_probability_is_zero(self):
        crossover = UnimodalNormalDistributionCrossover(0.0, 0.5, 0.35)
        solution1 = FloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = FloatSolution([1, 2], [2, 4], 2, 2)
        solution3 = FloatSolution([1, 2], [2, 4], 2, 2)
        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]
        solution3.variables = [1.9, 3.2]
        
        offspring = crossover.execute([solution1, solution2, solution3])
        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)

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

        solution1 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution2 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution3 = NewFloatSolution([1, 2], [2, 4], 2, 2)
        solution1.variables = [1.5, 2.7]
        solution2.variables = [1.7, 3.6]
        solution3.variables = [1.9, 3.2]
        
        crossover = UnimodalNormalDistributionCrossover(0.0, 0.5, 0.35)
        offspring = crossover.execute([solution1, solution2, solution3])
        
        self.assertEqual(2, len(offspring))
        self.assertEqual(solution1.variables, offspring[0].variables)
        self.assertEqual(solution2.variables, offspring[1].variables)
        self.assertIsInstance(offspring[0], NewFloatSolution)
        self.assertIsInstance(offspring[1], NewFloatSolution)

    @mock.patch('random.random')
    @mock.patch('random.uniform')
    def test_should_execute_perform_undx_crossover(self, uniform_mock, random_mock):
        # Mock random.random() for probability check (0.05 < 0.9, so crossover happens)
        # and for beta calculation (two calls with 0.3 and 0.5)
        random_mock.side_effect = [0.05, 0.3, 0.5, 0.3, 0.5]  # Add extra values for the second variable
        
        # Mock random.uniform() for alpha values (zeta=0.5, distance=1.0, so alpha in [-0.5, 0.5])
        # Return 0.2 for both variables
        uniform_mock.return_value = 0.2
        
        crossover = UnimodalNormalDistributionCrossover(0.9, 0.5, 0.35)
        
        # Create three parent solutions with known values
        solution1 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution2 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution3 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        
        # Set parent values (distance between p1 and p2 is 1.0)
        solution1.variables = [2.0, 3.0]
        solution2.variables = [3.0, 3.0]
        solution3.variables = [2.5, 4.0]  # Used for orthogonal direction
        
        # Expected calculations:
        # Center = [(2+3)/2, (3+3)/2] = [2.5, 3.0]
        # diff = [3-2, 3-3] = [1.0, 0.0]
        # distance = 1.0
        # alpha = 0.2 (mocked)
        # For first variable (i=0):
        #   beta = (0.3-0.5 + 0.5-0.5) * 0.35 * 1.0 = (-0.2 + 0.0) * 0.35 = -0.07
        #   orthogonal = (2.5-2.5)/1.0 = 0.0
        #   value1 = 2.5 + 0.2*1.0 + (-0.07)*0.0 = 2.5 + 0.2 + 0.0 = 2.7
        #   value2 = 2.5 - 0.2*1.0 - (-0.07)*0.0 = 2.5 - 0.2 - 0.0 = 2.3
        # 
        # For second variable (i=1):
        #   beta = (0.3-0.5 + 0.5-0.5) * 0.35 * 1.0 = (-0.2 + 0.0) * 0.35 = -0.07
        #   orthogonal = (4.0-3.0)/1.0 = 1.0
        #   value1 = 3.0 + 0.2*0.0 + (-0.07)*1.0 = 3.0 + 0.0 - 0.07 = 2.93
        #   value2 = 3.0 - 0.2*0.0 - (-0.07)*1.0 = 3.0 - 0.0 + 0.07 = 3.07
        
        offspring = crossover.execute([solution1, solution2, solution3])
        
        self.assertEqual(2, len(offspring))
        
        # Check the values with a small tolerance due to floating-point arithmetic
        self.assertAlmostEqual(offspring[0].variables[0], 2.7, places=6)
        self.assertAlmostEqual(offspring[0].variables[1], 2.93, places=6)
        self.assertAlmostEqual(offspring[1].variables[0], 2.3, places=6)
        self.assertAlmostEqual(offspring[1].variables[1], 3.07, places=6)
        
        # Verify random.random() was called for probability and beta calculation (1 for probability + 4 for beta)
        self.assertEqual(random_mock.call_count, 5)
        # Verify random.uniform() was called for each variable (2 variables)
        self.assertEqual(uniform_mock.call_count, 2)

    def test_should_handle_identical_parents_gracefully(self):
        # Test case where parent1 and parent2 are identical
        crossover = UnimodalNormalDistributionCrossover(1.0, 0.5, 0.35)
        
        solution1 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution2 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        solution3 = FloatSolution([1.0, 1.0], [5.0, 5.0], 1, 0)
        
        solution1.variables = [2.0, 3.0]
        solution2.variables = [2.0, 3.0]  # Same as solution1
        solution3.variables = [2.5, 4.0]
        
        # This should not raise an exception
        try:
            offspring = crossover.execute([solution1, solution2, solution3])
            self.assertEqual(2, len(offspring))
        except Exception as e:
            self.fail(f"Should handle identical parents but got: {e}")


if __name__ == "__main__":
    unittest.main()
