"""
Unit tests for normalization utilities.

This module contains comprehensive tests for the normalization functions
to ensure they work correctly across different scenarios and edge cases.
"""

import unittest
import numpy as np
from unittest import TestCase

from jmetal.util.normalization import (
    normalize_fronts,
    normalize_front, 
    get_ideal_and_nadir_points,
    normalize_to_unit_hypercube,
    solutions_to_matrix
)

# Test tolerance for numerical comparisons (matching Julia implementation)
NORMALIZATION_TEST_ATOL = 1e-12


class TestNormalizationUtilities(TestCase):
    """Test cases for normalization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard test data
        self.front = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 150.0]])
        self.reference_front = np.array([[0.5, 80.0], [1.5, 120.0], [2.5, 180.0]])
        
        # Test data with known statistical properties
        self.front_stats = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        self.reference_stats = np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]])
    
    def test_normalize_fronts_reference_only(self):
        """Test reference_only normalization method."""
        norm_front, norm_ref = normalize_fronts(
            self.front, self.reference_front, method="reference_only"
        )
        
        # Reference front should be normalized to [0,1] range
        ref_min = np.min(norm_ref, axis=0)
        ref_max = np.max(norm_ref, axis=0)
        
        np.testing.assert_allclose(ref_min, [0.0, 0.0], atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(ref_max, [1.0, 1.0], atol=NORMALIZATION_TEST_ATOL)
        
        # Check shapes are preserved
        self.assertEqual(norm_front.shape, self.front.shape)
        self.assertEqual(norm_ref.shape, self.reference_front.shape)
    
    def test_normalize_fronts_reference_only_detailed(self):
        """Test reference_only method with specific expected values."""
        front = np.array([[0.0, 2.0], [4.0, 6.0]])
        reference_front = np.array([[1.0, 3.0], [3.0, 5.0]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Reference bounds: obj1 [1,3], obj2 [3,5]
        # Expected normalized reference: [[0,0], [1,1]]
        expected_norm_ref = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(norm_ref, expected_norm_ref, atol=NORMALIZATION_TEST_ATOL)
        
        # Expected normalized front: [[-0.5, -0.5], [1.5, 1.5]]
        expected_norm_front = np.array([[-0.5, -0.5], [1.5, 1.5]])
        np.testing.assert_allclose(norm_front, expected_norm_front, atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalize_fronts_minmax(self):
        """Test minmax normalization method."""
        front = np.array([[1.0, 5.0], [3.0, 7.0]])
        reference_front = np.array([[2.0, 6.0], [4.0, 8.0]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="minmax")
        
        # Global bounds: obj1 [1,4], obj2 [5,8]
        # All values should be in [0,1]
        self.assertTrue(np.all(norm_front >= 0.0))
        self.assertTrue(np.all(norm_front <= 1.0))
        self.assertTrue(np.all(norm_ref >= 0.0))
        self.assertTrue(np.all(norm_ref <= 1.0))
        
        # Check specific values
        expected_front = np.array([[0.0, 0.0], [2/3, 2/3]])
        expected_ref = np.array([[1/3, 1/3], [1.0, 1.0]])
        np.testing.assert_allclose(norm_front, expected_front, atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(norm_ref, expected_ref, atol=NORMALIZATION_TEST_ATOL)
        
        # Combined range should be normalized to [0,1]
        combined_norm = np.vstack([norm_front, norm_ref])
        global_min = np.min(combined_norm, axis=0)
        global_max = np.max(combined_norm, axis=0)
        
        np.testing.assert_allclose(global_min, [0.0, 0.0], atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(global_max, [1.0, 1.0], atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalize_fronts_zscore(self):
        """Test zscore normalization method."""
        front = np.array([[1.0, 1.0], [2.0, 2.0]])
        reference_front = np.array([[3.0, 3.0], [4.0, 4.0]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="zscore")
        
        # Combined normalized data should have approximately zero mean
        combined_norm = np.vstack([norm_front, norm_ref])
        combined_mean = np.mean(combined_norm, axis=0)
        
        np.testing.assert_allclose(combined_mean, [0.0, 0.0], atol=NORMALIZATION_TEST_ATOL)
        
        # Standard deviation should be approximately 1
        combined_std = np.std(combined_norm, axis=0, ddof=1)
        np.testing.assert_allclose(combined_std, [1.0, 1.0], atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalize_front_with_bounds(self):
        """Test normalize_front with predefined bounds."""
        front = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 150.0]])
        bounds_min = np.array([0.0, 50.0])
        bounds_max = np.array([5.0, 250.0])
        
        normalized = normalize_front(front, bounds_min, bounds_max)
        
        # Check normalization formula: (x - min) / (max - min)
        expected = np.array([[0.2, 0.25], [0.4, 0.75], [0.6, 0.5]])
        
        np.testing.assert_allclose(normalized, expected, atol=NORMALIZATION_TEST_ATOL)
        
        # All values should be in reasonable range for this test case
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
    
    def test_normalize_front_zero_range(self):
        """Test normalize_front with zero range in bounds."""
        front = np.array([[1.0, 2.0], [3.0, 4.0]])
        bounds_min = np.array([0.0, 3.0])
        bounds_max = np.array([2.0, 3.0])  # Second objective has zero range
        
        normalized = normalize_front(front, bounds_min, bounds_max)
        
        # First objective should normalize properly: (x - 0) / (2 - 0)
        expected_obj1 = np.array([0.5, 1.5])
        np.testing.assert_allclose(normalized[:, 0], expected_obj1, atol=NORMALIZATION_TEST_ATOL)
        
        # Second objective should just subtract the constant: x - 3
        expected_obj2 = np.array([-1.0, 1.0])
        np.testing.assert_allclose(normalized[:, 1], expected_obj2, atol=NORMALIZATION_TEST_ATOL)
    
    def test_ideal_and_nadir_points(self):
        """Test ideal and nadir point calculation."""
        front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        
        ideal, nadir = get_ideal_and_nadir_points(front)
        
        expected_ideal = np.array([1.0, 1.0])
        expected_nadir = np.array([3.0, 3.0])
        
        np.testing.assert_allclose(ideal, expected_ideal, atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(nadir, expected_nadir, atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalize_to_unit_hypercube(self):
        """Test normalization to unit hypercube."""
        front = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        
        normalized = normalize_to_unit_hypercube(front)
        
        # Should be in [0,1] range
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
        
        # Min should be 0, max should be 1
        np.testing.assert_allclose(np.min(normalized, axis=0), [0.0, 0.0], atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(np.max(normalized, axis=0), [1.0, 1.0], atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalize_to_unit_hypercube_with_custom_bounds(self):
        """Test unit hypercube normalization with custom ideal/nadir points."""
        front = np.array([[2.0, 4.0], [3.0, 3.0], [4.0, 2.0]])
        ideal = np.array([1.0, 1.0])
        nadir = np.array([5.0, 5.0])
        
        normalized = normalize_to_unit_hypercube(front, ideal, nadir)
        
        # Should normalize using custom bounds
        expected = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
        np.testing.assert_allclose(normalized, expected, atol=NORMALIZATION_TEST_ATOL)
    
    def test_constant_objective_values(self):
        """Test handling of constant objective values."""
        # Front with constant values in one objective
        front = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        reference_front = np.array([[1.0, 5.0], [2.0, 5.0]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Should handle constant values gracefully (no NaN or Inf)
        self.assertFalse(np.any(np.isnan(norm_front)))
        self.assertFalse(np.any(np.isnan(norm_ref)))
        self.assertFalse(np.any(np.isinf(norm_front)))
        self.assertFalse(np.any(np.isinf(norm_ref)))
        
    def test_identical_fronts(self):
        """Test normalization with identical fronts."""
        identical_front = np.array([[1.0, 2.0], [3.0, 4.0]])
        norm_front, norm_ref = normalize_fronts(identical_front, identical_front, method="reference_only")
        
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(norm_front, expected, atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(norm_ref, expected, atol=NORMALIZATION_TEST_ATOL)
    
    def test_constant_column_in_reference(self):
        """Test handling of constant column in reference front."""
        front = np.array([[1.0, 2.0], [2.0, 3.0]])
        reference_front = np.array([[1.5, 2.5], [1.5, 2.5]])  # First column is constant
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # First column should be centered around 0 (constant removed)
        expected_front_obj1 = np.array([-0.5, 0.5])
        expected_ref_obj1 = np.array([0.0, 0.0])
        
        np.testing.assert_allclose(norm_front[:, 0], expected_front_obj1, atol=NORMALIZATION_TEST_ATOL)
        np.testing.assert_allclose(norm_ref[:, 0], expected_ref_obj1, atol=NORMALIZATION_TEST_ATOL)
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        empty_front = np.array([]).reshape(0, 2)
        
        ideal, nadir = get_ideal_and_nadir_points(empty_front)
        
        self.assertEqual(ideal.size, 0)
        self.assertEqual(nadir.size, 0)
        
        # Unit hypercube normalization of empty array
        normalized = normalize_to_unit_hypercube(empty_front)
        self.assertEqual(normalized.shape, (0, 2))
    
    def test_single_point_front(self):
        """Test handling of single-point fronts."""
        single_front = np.array([[1.0, 2.0]])
        single_ref = np.array([[0.5, 1.5]])
        
        norm_front, norm_ref = normalize_fronts(single_front, single_ref, method="reference_only")
        
        # Should work without errors
        self.assertEqual(norm_front.shape, (1, 2))
        self.assertEqual(norm_ref.shape, (1, 2))
    
    def test_dimension_mismatch_error(self):
        """Test error handling for dimension mismatches."""
        front = np.array([[1.0, 2.0]])
        reference_front = np.array([[1.0, 2.0, 3.0]])  # Different number of objectives
        
        with self.assertRaises(ValueError):
            normalize_fronts(front, reference_front)
        
        # Test bounds dimension mismatch
        bounds_min = np.array([0.0])
        bounds_max = np.array([1.0, 2.0])  # Different lengths
        
        with self.assertRaises(ValueError):
            normalize_front(front, bounds_min, bounds_max)
    
    def test_invalid_normalization_method(self):
        """Test error handling for invalid normalization methods."""
        with self.assertRaises(ValueError):
            normalize_fronts(self.front, self.reference_front, method="invalid_method")
    
    def test_solutions_to_matrix_with_mock_solutions(self):
        """Test solutions_to_matrix with mock Solution objects."""
        # Create mock solution objects
        class MockSolution:
            def __init__(self, objectives):
                self.objectives = objectives
        
        solutions = [
            MockSolution([1.0, 2.0]),
            MockSolution([3.0, 4.0]),
            MockSolution([5.0, 6.0])
        ]
        
        matrix = solutions_to_matrix(solutions)
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        np.testing.assert_allclose(matrix, expected)
    
    def test_solutions_to_matrix_empty_list(self):
        """Test solutions_to_matrix with empty list."""
        matrix = solutions_to_matrix([])
        self.assertEqual(matrix.shape, (0, 0))
    
    def test_three_objectives_normalization(self):
        """Test normalization with three objectives."""
        front_3d = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
        reference_3d = np.array([[0.5, 5.0, 50.0], [1.5, 15.0, 150.0]])
        
        norm_front, norm_ref = normalize_fronts(front_3d, reference_3d, method="reference_only")
        
        # Check all objectives are properly normalized
        self.assertEqual(norm_front.shape, (2, 3))
        self.assertEqual(norm_ref.shape, (2, 3))
        
        # Reference should span [0,1] for each objective
        for obj in range(3):
            np.testing.assert_allclose(np.min(norm_ref[:, obj]), 0.0, atol=NORMALIZATION_TEST_ATOL)
            np.testing.assert_allclose(np.max(norm_ref[:, obj]), 1.0, atol=NORMALIZATION_TEST_ATOL)
    
    def test_mathematical_properties_preservation(self):
        """Test that relative ordering and distance ratios are preserved."""
        front = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        reference_front = np.array([[0.5, 5.0], [1.5, 15.0], [2.5, 25.0]])
        
        # Test that relative ordering is preserved
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Original ordering in first objective: 1 < 2 < 3
        self.assertTrue(norm_front[0, 0] < norm_front[1, 0] < norm_front[2, 0])
        # Original ordering in second objective: 10 < 20 < 30
        self.assertTrue(norm_front[0, 1] < norm_front[1, 1] < norm_front[2, 1])
        
        # Test that distance ratios are preserved (for reference_only method)
        original_diff_obj1_12 = front[1, 0] - front[0, 0]  # 2 - 1 = 1
        original_diff_obj1_23 = front[2, 0] - front[1, 0]  # 3 - 2 = 1
        
        normalized_diff_obj1_12 = norm_front[1, 0] - norm_front[0, 0]
        normalized_diff_obj1_23 = norm_front[2, 0] - norm_front[1, 0]
        
        # Ratios should be the same since differences are equal
        np.testing.assert_allclose(normalized_diff_obj1_12, normalized_diff_obj1_23, 
                                  atol=NORMALIZATION_TEST_ATOL)
    
    def test_normalization_preserves_pareto_dominance(self):
        """Test that normalization preserves Pareto dominance relationships."""
        # Create a front where solution 1 dominates solution 2
        front = np.array([[1.0, 2.0], [2.0, 3.0]])  # (1,2) dominates (2,3)
        reference_front = np.array([[0.0, 0.0], [3.0, 4.0]])
        
        norm_front, _ = normalize_fronts(front, reference_front, method="reference_only")
        
        # After normalization, dominance should be preserved
        # norm_front[0] should still dominate norm_front[1]
        self.assertTrue(np.all(norm_front[0] <= norm_front[1]))
        self.assertTrue(np.any(norm_front[0] < norm_front[1]))


class TestNormalizationEdgeCases(TestCase):
    """Test edge cases and special scenarios."""
    
    def test_very_large_values(self):
        """Test normalization with very large values."""
        front = np.array([[1e6, 1e9], [2e6, 2e9]])
        reference_front = np.array([[0, 0], [3e6, 3e9]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Should handle large values without overflow
        self.assertFalse(np.any(np.isnan(norm_front)))
        self.assertFalse(np.any(np.isinf(norm_front)))
    
    def test_very_small_values(self):
        """Test normalization with very small values."""
        front = np.array([[1e-9, 1e-12], [2e-9, 2e-12]])
        reference_front = np.array([[0, 0], [3e-9, 3e-12]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Should handle small values without underflow
        self.assertFalse(np.any(np.isnan(norm_front)))
        self.assertFalse(np.any(np.isinf(norm_front)))
    
    def test_negative_values(self):
        """Test normalization with negative values."""
        front = np.array([[-1.0, -2.0], [0.0, -1.0], [1.0, 0.0]])
        reference_front = np.array([[-2.0, -3.0], [2.0, 1.0]])
        
        norm_front, norm_ref = normalize_fronts(front, reference_front, method="reference_only")
        
        # Should handle negative values correctly
        self.assertFalse(np.any(np.isnan(norm_front)))
        self.assertFalse(np.any(np.isinf(norm_front)))


if __name__ == '__main__':
    unittest.main()