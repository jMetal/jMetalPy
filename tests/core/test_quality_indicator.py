"""
Test cases for quality indicators in jMetalPy.

This module contains comprehensive test cases for quality indicators including:
- InvertedGenerationalDistance (IGD) with power parameter support
- InvertedGenerationalDistancePlus (IGD+) with dominance-based distance
- AdditiveEpsilonIndicator with comprehensive mathematical properties
- Legacy indicators for backwards compatibility

Test cases are inspired by the Julia implementation in MetaJul to ensure
mathematical correctness and comprehensive coverage of edge cases.
"""
import os
import unittest
from pathlib import Path

import numpy as np

from jmetal.core.quality_indicator import (EpsilonIndicator, AdditiveEpsilonIndicator,
                                           GenerationalDistance, HyperVolume,
                                           InvertedGenerationalDistance,
                                           InvertedGenerationalDistancePlus,
                                           NormalizedHyperVolume)

DIRNAME = os.path.dirname(os.path.abspath(__file__))
EPSILON_TEST_ATOL = 1e-12


class GenerationalDistanceTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = GenerationalDistance([])
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Generational Distance", GenerationalDistance([]).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("GD", GenerationalDistance([]).get_short_name())

    def test_case1(self):
        """
        Case 1. Reference front: [[1.0, 1.0]], front: [[1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case2(self):
        """
        Case 2. Reference front: [[1.0, 1.0], [2.0, 2.0], front: [[1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case3(self):
        """
        Case 3. Reference front: [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], front: [[1.0, 1.0, 1.0]]
        Expected result: the distance to the nearest point of the reference front is 0.0. Example with three objectives

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
        front = np.array([[1.0, 1.0, 1.0]])

        result = indicator.compute(front)

        self.assertEqual(0.0, result)

    def test_case4(self):
        """
        Case 4. reference front: [[1.0, 1.0], [2.0, 2.0]], front: [[1.5, 1.5]]
        Expected result: the distance to the nearest point of the reference front is the euclidean distance to any of the
        points of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.5, 1.5]])

        result = indicator.compute(front)

        self.assertEqual(np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2)), result)
        self.assertEqual(np.sqrt(pow(2.0 - 1.5, 2) + pow(2.0 - 1.5, 2)), result)

    def test_case5(self):
        """
        Case 5. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5]]
        Expected result: the distance to the nearest point of the reference front is the euclidean distance
        to the nearest point of the reference front ([1.0, 1.0])

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5]])

        result = indicator.compute(front)

        self.assertEqual(np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2)), result)
        self.assertEqual(np.sqrt(pow(2.0 - 1.5, 2) + pow(2.0 - 1.5, 2)), result)

    def test_case6(self):
        """
        Case 6. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2]]
        Expected result: the distance to the nearest point of the reference front is the average of the sum of each point
        of the front to the nearest point of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case7(self):
        """
        Case 7. reference front: [[1.0, 1.0], [2.1, 2.1]], front: [[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]]
        Expected result: the distance to the nearest point of the reference front is the sum of each point of the front to the
        nearest point of the reference front

        :return:
        """
        indicator = GenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]])

        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))
        distance_of_third_point = np.sqrt(pow(2.1 - 1.9, 2) + pow(2.1 - 1.9, 2))

        self.assertEqual((distance_of_first_point + distance_of_second_point + distance_of_third_point) / 3.0, result)


class InvertedGenerationalDistanceTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0]]))
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Inverted Generational Distance", InvertedGenerationalDistance(np.array([[1.0, 1.0]])).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("IGD", InvertedGenerationalDistance(np.array([[1.0, 1.0]])).get_short_name())

    def test_identical_fronts_should_return_zero(self):
        """Identical 2D fronts: IGD = 0"""
        identical_fronts = np.array([[0.1, 0.2], [0.3, 0.4]])
        identical_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = InvertedGenerationalDistance(identical_reference)
        result = indicator.compute(identical_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_uniform_shift_2d(self):
        """Uniform shift in 2D using corrected IGD formula (L2 norm)"""
        shifted_fronts = np.array([[0.2, 0.3], [0.4, 0.5]])
        shifted_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = InvertedGenerationalDistance(shifted_reference)
        result = indicator.compute(shifted_fronts)
        # Using L2 norm formula: sqrt(sum(d²))/N
        # Each distance = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) ≈ 0.1414213562373095
        # IGD = sqrt(2 * 0.02) / 2 = sqrt(0.04) / 2 = 0.2 / 2 = 0.1
        expected = 0.1
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_single_point_fronts_2d(self):
        """Single-point fronts in 2D"""
        single_front = np.array([[0.5, 0.5]])
        single_reference = np.array([[0.2, 0.3]])
        indicator = InvertedGenerationalDistance(single_reference)
        result = indicator.compute(single_front)
        expected = np.sqrt((0.5-0.2)**2 + (0.5-0.3)**2)
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_three_objective_identical_fronts(self):
        """Three-objective fronts: IGD = 0"""
        three_obj_fronts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        three_obj_reference = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        indicator = InvertedGenerationalDistance(three_obj_reference)
        result = indicator.compute(three_obj_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_shifted_three_objective_fronts(self):
        """Shifted three-objective fronts using corrected IGD formula"""
        shifted_three_obj_fronts = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        shifted_three_obj_reference = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        indicator = InvertedGenerationalDistance(shifted_three_obj_reference)
        result = indicator.compute(shifted_three_obj_fronts)
        # Each reference point distance = sqrt(3 * 0.1^2) = sqrt(0.03) = 0.1732...
        # IGD = sqrt(sum(d²))/N = sqrt(0.03 + 0.03) / 2 = sqrt(0.06) / 2 = 0.1224...
        expected = np.sqrt(2 * 3 * 0.1**2) / 2
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_perfect_match_2d(self):
        """Perfect match in 2D → IGD = 0"""
        perfect_match_fronts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        perfect_match_reference = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        indicator = InvertedGenerationalDistance(perfect_match_reference)
        result = indicator.compute(perfect_match_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_uniformly_shifted_solutions_2d(self):
        """Uniformly shifted solutions in 2D using corrected IGD formula"""
        uniform_shifted_fronts = np.array([[0.1, 0.1], [0.6, 0.6], [1.1, 1.1]])
        uniform_shifted_reference = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        indicator = InvertedGenerationalDistance(uniform_shifted_reference)
        result = indicator.compute(uniform_shifted_fronts)
        # Distance for each point = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) = 0.1414...
        # IGD = sqrt(3 * 0.02) / 3 = sqrt(0.06) / 3 = 0.08164...
        expected = np.sqrt(3 * 0.02) / 3
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_partial_coverage_2d(self):
        """Partial coverage of the front in 2D using corrected IGD formula"""
        partial_coverage_fronts = np.array([[0.0, 0.0], [1.0, 0.0]])
        partial_coverage_reference = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        indicator = InvertedGenerationalDistance(partial_coverage_reference)
        result = indicator.compute(partial_coverage_fronts)
        # Distances: 0, 1, 0, 1 using L2 norm formula: sqrt(sum(d²))/N
        # IGD = sqrt(0² + 1² + 0² + 1²) / 4 = sqrt(2) / 4 = 1.414... / 4 = 0.3535...
        expected = np.sqrt(2) / 4
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_sparse_approximation_2d(self):
        """Sparse approximation of continuous front in 2D using corrected IGD formula"""
        sparse_fronts = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        sparse_reference = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
        indicator = InvertedGenerationalDistance(sparse_reference)
        result = indicator.compute(sparse_fronts)
        # Using L2 norm formula: sqrt(sum(d²))/N
        # Distances: 0, sqrt((0.25-0.5)^2 + (0.75-0.5)^2), 0, sqrt((0.75-0.5)^2 + (0.25-0.5)^2), 0
        # = 0, sqrt(0.0625 + 0.0625), 0, sqrt(0.0625 + 0.0625), 0
        # = 0, sqrt(0.125), 0, sqrt(0.125), 0 = 0, 0.3535..., 0, 0.3535..., 0
        # IGD = sqrt(0 + 0.125 + 0 + 0.125 + 0) / 5 = sqrt(0.25) / 5 = 0.5 / 5 = 0.1
        distance_middle = np.sqrt((0.25-0.5)**2 + (0.75-0.5)**2)  # For [0.25, 0.75] and [0.75, 0.25]
        expected = np.sqrt(2 * distance_middle**2) / 5  # Using L2 norm formula
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_single_objective(self):
        """Single objective case"""
        single_obj_front = np.array([[2.0]])
        single_obj_reference = np.array([[1.0]])
        indicator = InvertedGenerationalDistance(single_obj_reference)
        result = indicator.compute(single_obj_front)
        self.assertAlmostEqual(1.0, result, delta=EPSILON_TEST_ATOL)

    def test_dimension_mismatch_should_raise_error(self):
        """Dimension mismatch should throw error"""
        front_2d = np.array([[0.1, 0.2]])
        reference_3d = np.array([[0.1, 0.2, 0.3]])
        indicator = InvertedGenerationalDistance(reference_3d)
        with self.assertRaises(ValueError):
            indicator.compute(front_2d)

    def test_empty_front_behavior(self):
        """Empty front behavior"""
        empty_front = np.array([]).reshape(0, 2)
        non_empty_reference = np.array([[0.1, 0.2]])
        indicator = InvertedGenerationalDistance(non_empty_reference)
        with self.assertRaises(ValueError):
            indicator.compute(empty_front)

    def test_none_reference_front_should_raise_error(self):
        """None reference front should raise error"""
        with self.assertRaises(ValueError):
            InvertedGenerationalDistance(None)

    def test_empty_reference_front_should_raise_error(self):
        """Empty reference front should raise error"""
        with self.assertRaises(ValueError):
            InvertedGenerationalDistance(np.array([]).reshape(0, 2))

    def test_igd_with_power_parameter(self):
        """Test IGD with different power parameters"""
        igd_fronts = np.array([[0.0, 1.0], [1.0, 0.0]])
        igd_reference = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        
        # pow=1: Taxicab distance, summed then divided by N
        # Distances: 1, 0, 0, 1
        # IGD = (1 + 0 + 0 + 1) / 4 = 0.5
        indicator_pow1 = InvertedGenerationalDistance(igd_reference, pow=1.0)
        result_pow1 = indicator_pow1.compute(igd_fronts)
        expected_pow1 = 0.5
        self.assertAlmostEqual(expected_pow1, result_pow1, delta=EPSILON_TEST_ATOL)
        
        # pow=2: L2 norm formula: sqrt(sum(d²))/N
        # Distances squared: 1, 0, 0, 1
        # IGD = sqrt(1 + 0 + 0 + 1) / 4 = sqrt(2) / 4
        indicator_pow2 = InvertedGenerationalDistance(igd_reference, pow=2.0)
        result_pow2 = indicator_pow2.compute(igd_fronts)
        expected_pow2 = np.sqrt(2) / 4
        self.assertAlmostEqual(expected_pow2, result_pow2, delta=EPSILON_TEST_ATOL)

    def test_case1(self):
        """Legacy test case 1 - kept for backwards compatibility"""
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0]]))
        front = np.array([[1.0, 1.0]])
        result = indicator.compute(front)
        self.assertEqual(0.0, result)

    def test_case2(self):
        """Legacy test case 2 - kept for backwards compatibility"""
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.0, 1.0]])
        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2))
        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case3(self):
        """Legacy test case 3 - kept for backwards compatibility"""
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
        front = np.array([[1.0, 1.0, 1.0]])
        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2) + pow(1.0 - 1.0, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2) + pow(2.0 - 1.0, 2))
        self.assertEqual((distance_of_first_point + distance_of_second_point) / 2.0, result)

    def test_case4(self):
        """Legacy test case 4 - updated for corrected IGD formula"""
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.1, 2.1]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2]])
        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.1 - 2.2, 2) + pow(2.1 - 2.2, 2))
        # Using corrected L2 norm formula: sqrt(sum(d²))/N
        expected = np.sqrt(distance_of_first_point**2 + distance_of_second_point**2) / 2.0
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_case5(self):
        """Legacy test case 5 - updated for corrected IGD formula"""
        indicator = InvertedGenerationalDistance(np.array([[1.0, 1.0], [2.0, 2.0]]))
        front = np.array([[1.5, 1.5], [2.2, 2.2], [1.9, 1.9]])
        result = indicator.compute(front)
        distance_of_first_point = np.sqrt(pow(1.0 - 1.5, 2) + pow(1.0 - 1.5, 2))
        distance_of_second_point = np.sqrt(pow(2.0 - 1.9, 2) + pow(2.0 - 1.9, 2))
        # Using corrected L2 norm formula: sqrt(sum(d²))/N
        expected = np.sqrt(distance_of_first_point**2 + distance_of_second_point**2) / 2.0
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)


class InvertedGenerationalDistancePlusTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = InvertedGenerationalDistancePlus(np.array([[1.0, 1.0]]))
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Inverted Generational Distance Plus", InvertedGenerationalDistancePlus(np.array([[1.0, 1.0]])).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("IGD+", InvertedGenerationalDistancePlus(np.array([[1.0, 1.0]])).get_short_name())

    def test_identical_fronts_should_return_zero(self):
        """Identical fronts: IGD+ = 0"""
        identical_fronts = np.array([[0.1, 0.2], [0.3, 0.4]])
        identical_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = InvertedGenerationalDistancePlus(identical_reference)
        result = indicator.compute(identical_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_uniform_shift_should_be_positive(self):
        """Uniform shift: IGD+ > 0"""
        shifted_fronts = np.array([[0.2, 0.3], [0.4, 0.5]])
        shifted_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = InvertedGenerationalDistancePlus(shifted_reference)
        result = indicator.compute(shifted_fronts)
        self.assertGreater(result, 0.0)

    def test_single_point_fronts_should_be_positive(self):
        """Single-point fronts"""
        single_front = np.array([[0.5, 0.5]])
        single_reference = np.array([[0.2, 0.3]])
        indicator = InvertedGenerationalDistancePlus(single_reference)
        result = indicator.compute(single_front)
        self.assertGreater(result, 0.0)

    def test_three_objective_identical_fronts(self):
        """Three-objective fronts: IGD+ = 0"""
        three_obj_fronts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        three_obj_reference = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        indicator = InvertedGenerationalDistancePlus(three_obj_reference)
        result = indicator.compute(three_obj_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_non_negativity(self):
        """IGD+ should always be non-negative"""
        front_a = np.array([[0.0, 1.0], [1.0, 0.0]])
        front_b = np.array([[0.5, 0.5]])
        indicator_a = InvertedGenerationalDistancePlus(front_b)
        indicator_b = InvertedGenerationalDistancePlus(front_a)
        
        result_a = indicator_a.compute(front_a)
        result_b = indicator_b.compute(front_b)
        
        self.assertGreaterEqual(result_a, 0.0)
        self.assertGreaterEqual(result_b, 0.0)

    def test_asymmetry(self):
        """Test asymmetry with very different sized fronts"""
        single_point_front = np.array([[0.4, 0.6]])
        large_reference = np.array([
            [0.0, 0.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.5, 0.5],
            [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 1.0]
        ])
        
        indicator_single_to_large = InvertedGenerationalDistancePlus(large_reference)
        indicator_large_to_single = InvertedGenerationalDistancePlus(single_point_front)
        
        igdplus_single_to_large = indicator_single_to_large.compute(single_point_front)
        igdplus_large_to_single = indicator_large_to_single.compute(large_reference)
        
        # With 1 vs 10 points and no exact match, these should be different
        self.assertNotEqual(igdplus_single_to_large, igdplus_large_to_single)
        self.assertGreater(igdplus_single_to_large, 0.0)
        self.assertGreaterEqual(igdplus_large_to_single, 0.0)

    def test_single_objective(self):
        """Single objective case"""
        single_obj_front = np.array([[2.0]])
        single_obj_reference = np.array([[1.0]])
        indicator = InvertedGenerationalDistancePlus(single_obj_reference)
        result = indicator.compute(single_obj_front)
        self.assertGreaterEqual(result, 0.0)

    def test_dimension_mismatch_should_raise_error(self):
        """Dimension mismatch should throw error"""
        front_2d = np.array([[0.1, 0.2]])
        reference_3d = np.array([[0.1, 0.2, 0.3]])
        indicator = InvertedGenerationalDistancePlus(reference_3d)
        with self.assertRaises(ValueError):
            indicator.compute(front_2d)

    def test_empty_front_behavior(self):
        """Empty front behavior"""
        empty_front = np.array([]).reshape(0, 2)
        non_empty_reference = np.array([[0.1, 0.2]])
        indicator = InvertedGenerationalDistancePlus(non_empty_reference)
        with self.assertRaises(ValueError):
            indicator.compute(empty_front)

    def test_none_reference_front_should_raise_error(self):
        """None reference front should raise error"""
        with self.assertRaises(ValueError):
            InvertedGenerationalDistancePlus(None)

    def test_empty_reference_front_should_raise_error(self):
        """Empty reference front should raise error"""
        with self.assertRaises(ValueError):
            InvertedGenerationalDistancePlus(np.array([]).reshape(0, 2))


class AdditiveEpsilonIndicatorTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = AdditiveEpsilonIndicator(np.array([[1.0, 1.0], [2.0, 2.0]]))
        self.assertIsNotNone(indicator)

    def test_get_name_return_the_right_value(self):
        self.assertEqual("Additive Epsilon", AdditiveEpsilonIndicator(np.array([[1.0, 1.0]])).get_name())

    def test_get_short_name_return_the_right_value(self):
        self.assertEqual("EP", AdditiveEpsilonIndicator(np.array([[1.0, 1.0]])).get_short_name())

    def test_identical_fronts_should_return_zero(self):
        """Identical fronts: epsilon should be 0"""
        identical_fronts = np.array([[0.1, 0.2], [0.3, 0.4]])
        identical_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = AdditiveEpsilonIndicator(identical_reference)
        result = indicator.compute(identical_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_uniform_shift(self):
        """Uniform shift: front is worse by 0.1 in all objectives, epsilon should be 0.1"""
        shifted_fronts = np.array([[0.2, 0.3], [0.4, 0.5]])
        shifted_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = AdditiveEpsilonIndicator(shifted_reference)
        result = indicator.compute(shifted_fronts)
        self.assertAlmostEqual(0.1, result, delta=EPSILON_TEST_ATOL)

    def test_mixed_dominance(self):
        """Mixed dominance: only one solution in front dominates reference, epsilon should be 0.0"""
        mixed_fronts = np.array([[0.1, 0.2], [0.5, 0.6]])
        mixed_reference = np.array([[0.1, 0.2], [0.3, 0.4]])
        indicator = AdditiveEpsilonIndicator(mixed_reference)
        result = indicator.compute(mixed_fronts)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)

    def test_reference_point_outside_front(self):
        """Reference point outside front: epsilon should be positive"""
        outside_fronts = np.array([[0.2, 0.2], [0.3, 0.3]])
        outside_reference = np.array([[0.1, 0.1]])
        indicator = AdditiveEpsilonIndicator(outside_reference)
        result = indicator.compute(outside_fronts)
        self.assertAlmostEqual(0.1, result, delta=EPSILON_TEST_ATOL)

    def test_single_point_fronts(self):
        """Single-point fronts"""
        single_front = np.array([[0.5, 0.5]])
        single_reference = np.array([[0.2, 0.3]])
        indicator = AdditiveEpsilonIndicator(single_reference)
        result = indicator.compute(single_front)
        expected = max(0.5 - 0.2, 0.5 - 0.3)  # max(0.3, 0.2) = 0.3
        self.assertAlmostEqual(expected, result, delta=EPSILON_TEST_ATOL)

    def test_simple_2d_case(self):
        """Simple 2D case: epsilon = 1.0"""
        simple_front = np.array([[2.0, 3.0]])
        simple_reference = np.array([[1.0, 2.0]])
        indicator = AdditiveEpsilonIndicator(simple_reference)
        result = indicator.compute(simple_front)
        self.assertAlmostEqual(1.0, result, delta=EPSILON_TEST_ATOL)

    def test_three_point_fronts(self):
        """2D, three-point fronts: epsilon = 1.0"""
        three_point_fronts = np.array([[1.5, 4.0], [2.0, 3.0], [3.0, 2.0]])
        three_point_reference = np.array([[1.0, 3.0], [1.5, 2.0], [2.0, 1.5]])
        indicator = AdditiveEpsilonIndicator(three_point_reference)
        result = indicator.compute(three_point_fronts)
        self.assertAlmostEqual(1.0, result, delta=EPSILON_TEST_ATOL)

    def test_partial_three_fronts(self):
        """2D, three-point fronts: epsilon = 0.5"""
        partial_three_fronts = np.array([[1.5, 4.0], [1.5, 2.0], [2.0, 1.5]])
        partial_three_reference = np.array([[1.0, 3.0], [1.5, 2.0], [2.0, 1.5]])
        indicator = AdditiveEpsilonIndicator(partial_three_reference)
        result = indicator.compute(partial_three_fronts)
        self.assertAlmostEqual(0.5, result, delta=EPSILON_TEST_ATOL)

    def test_single_objective(self):
        """Single objective case"""
        single_obj_front = np.array([[2.0]])
        single_obj_reference = np.array([[1.0]])
        indicator = AdditiveEpsilonIndicator(single_obj_reference)
        result = indicator.compute(single_obj_front)
        self.assertAlmostEqual(1.0, result, delta=EPSILON_TEST_ATOL)

    def test_dimension_mismatch_should_raise_error(self):
        """Dimension mismatch should throw error"""
        front_2d = np.array([[0.1, 0.2]])
        reference_3d = np.array([[0.1, 0.2, 0.3]])
        indicator = AdditiveEpsilonIndicator(reference_3d)
        with self.assertRaises(ValueError):
            indicator.compute(front_2d)

    def test_empty_front_should_raise_error(self):
        """Empty front should throw error"""
        empty_front = np.array([]).reshape(0, 2)
        non_empty_reference = np.array([[0.1, 0.2]])
        indicator = AdditiveEpsilonIndicator(non_empty_reference)
        with self.assertRaises(ValueError):
            indicator.compute(empty_front)

    def test_none_reference_front_should_raise_error(self):
        """None reference front should raise error"""
        with self.assertRaises(ValueError):
            AdditiveEpsilonIndicator(None)

    def test_empty_reference_front_should_raise_error(self):
        """Empty reference front should raise error"""
        with self.assertRaises(ValueError):
            AdditiveEpsilonIndicator(np.array([]).reshape(0, 2))

    def test_non_negativity_with_allowance_for_dominating_fronts(self):
        """Non-negativity (but epsilon can be negative when dominating)"""
        front_a = np.array([[0.0, 1.0], [1.0, 0.0]])
        front_b = np.array([[0.5, 0.5]])
        
        indicator_a = AdditiveEpsilonIndicator(front_b)
        indicator_b = AdditiveEpsilonIndicator(front_a)
        
        eps_a_to_b = indicator_a.compute(front_a)
        eps_b_to_a = indicator_b.compute(front_b)
        
        # Allow reasonable negative values when dominating
        self.assertGreaterEqual(eps_a_to_b, -1.0)
        self.assertGreaterEqual(eps_b_to_a, 0.0)

    def test_asymmetry_property(self):
        """Test asymmetry with fronts where dominating front just covers the reference"""
        reference_front = np.array([[0.2, 0.2]])  # Single point
        covering_front = np.array([[0.2, 0.2], [0.1, 0.3], [0.3, 0.1]])  # Includes the reference point exactly
        
        indicator_covering_to_ref = AdditiveEpsilonIndicator(reference_front)
        indicator_ref_to_covering = AdditiveEpsilonIndicator(covering_front)
        
        eps_covering_to_ref = indicator_covering_to_ref.compute(covering_front)
        eps_ref_to_covering = indicator_ref_to_covering.compute(reference_front)
        
        self.assertAlmostEqual(eps_covering_to_ref, 0.0, delta=EPSILON_TEST_ATOL)  # Should be exactly 0
        self.assertGreater(eps_ref_to_covering, 0.0)  # Reverse should be positive
        self.assertNotEqual(eps_covering_to_ref, eps_ref_to_covering)  # Should be different

    def test_real_data_identical_fronts(self):
        """Real data: identical fronts should give epsilon = 0"""
        # Using a simple synthetic front since we may not have ZDT1.csv
        real_front = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        indicator = AdditiveEpsilonIndicator(real_front)
        result = indicator.compute(real_front)
        self.assertAlmostEqual(0.0, result, delta=EPSILON_TEST_ATOL)


class EpsilonIndicatorTestCases(unittest.TestCase):
    """Test cases for backwards compatibility alias"""
    def test_should_constructor_create_a_non_null_object(self) -> None:
        indicator = EpsilonIndicator(np.array([[1.0, 1.0], [2.0, 2.0]]))
        self.assertIsNotNone(indicator)

    def test_epsilon_indicator_is_alias_for_additive_epsilon(self):
        """Test that EpsilonIndicator is an alias for AdditiveEpsilonIndicator"""
        reference = np.array([[1.0, 2.0], [2.0, 1.0]])
        front = np.array([[1.5, 1.5]])
        
        additive_indicator = AdditiveEpsilonIndicator(reference)
        epsilon_indicator = EpsilonIndicator(reference)
        
        result_additive = additive_indicator.compute(front)
        result_epsilon = epsilon_indicator.compute(front)
        
        self.assertEqual(result_additive, result_epsilon)
        self.assertEqual(additive_indicator.get_name(), epsilon_indicator.get_name())
        self.assertEqual(additive_indicator.get_short_name(), epsilon_indicator.get_short_name())


class HyperVolumeTestCases(unittest.TestCase):
    def test_should_hypervolume_return_5_0(self):
        reference_point = [2, 2, 2]

        front = np.array([[1, 0, 1], [0, 1, 0]])

        hv = HyperVolume(reference_point)
        value = hv.compute(front)

        self.assertEqual(5.0, value)

    def test_should_hypervolume_return_the_correct_value_when_applied_to_the_ZDT1_reference_front(self):
        filepath = Path(DIRNAME, "ZDT1.pf")
        front = []

        with open(filepath) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                front.append(vector)

        reference_point = [1, 1]

        hv = HyperVolume(reference_point)
        value = hv.compute(np.array(front))

        self.assertAlmostEqual(0.666, value, delta=0.001)


class NormalizedHyperVolumeTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = Path(DIRNAME, "ZDT1.pf")
        front = []

        with open(filepath) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                front.append(vector)

        cls._front = np.array(front)

    def test_should_hypervolume_return_zero_when_providing_reference_front(self):
        reference_point = [1, 1]
        reference_front = self._front

        hv = NormalizedHyperVolume(reference_point)
        hv.set_reference_front(reference_front)
        value = hv.compute(reference_front)

        self.assertAlmostEqual(0, value, delta=0.001)

    def test_should_raise_AssertionError_when_reference_front_hv_is_zero(self):
        reference_point = [0, 0]
        reference_front = self._front

        hv = NormalizedHyperVolume(reference_point)
        with self.assertRaises(AssertionError):
            hv.set_reference_front(reference_front)


if __name__ == "__main__":
    unittest.main()
