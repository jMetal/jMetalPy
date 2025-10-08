import unittest
import numpy as np

from jmetal.util.distance import DistanceMetric, DistanceCalculator


class DistanceMetricTestCase(unittest.TestCase):
    """Test cases for the new distance metrics and calculator in distance.py."""
    
    def setUp(self):
        self.point1 = np.array([0.0, 0.0, 0.0])
        self.point2 = np.array([1.0, 1.0, 1.0])
        self.point3 = np.array([0.5, 0.2, 0.8])
        
    def test_distance_metric_enum(self):
        """Test that DistanceMetric enum has expected values."""
        self.assertEqual(DistanceMetric.L2_SQUARED.value, "l2_squared")
        self.assertEqual(DistanceMetric.LINF.value, "linf")
        self.assertEqual(DistanceMetric.TCHEBY_WEIGHTED.value, "tcheby_weighted")
        
    def test_l2_squared_distance_calculation(self):
        """Test L2 squared distance calculation."""
        # Distance between (0,0,0) and (1,1,1) should be 3.0 (1² + 1² + 1²)
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point2, DistanceMetric.L2_SQUARED
        )
        self.assertAlmostEqual(3.0, distance, places=6)
        
        # Distance from point to itself should be 0
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point1, DistanceMetric.L2_SQUARED
        )
        self.assertAlmostEqual(0.0, distance, places=6)
        
    def test_linf_distance_calculation(self):
        """Test L-infinity (Chebyshev) distance calculation."""
        # L-infinity distance between (0,0,0) and (1,1,1) should be 1.0 (max difference)
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point2, DistanceMetric.LINF
        )
        self.assertAlmostEqual(1.0, distance, places=6)
        
        # Test with different point
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point3, DistanceMetric.LINF
        )
        # max(|0-0.5|, |0-0.2|, |0-0.8|) = 0.8
        self.assertAlmostEqual(0.8, distance, places=6)
        
    def test_tcheby_weighted_distance_calculation(self):
        """Test weighted Chebyshev distance calculation."""
        weights = np.array([2.0, 1.0, 0.5])
        
        # Weighted distance between (0,0,0) and (1,1,1) with weights [2,1,0.5]
        # max(2*|0-1|, 1*|0-1|, 0.5*|0-1|) = max(2, 1, 0.5) = 2.0
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point2, DistanceMetric.TCHEBY_WEIGHTED, weights
        )
        self.assertAlmostEqual(2.0, distance, places=6)
        
        # Test with equal weights
        equal_weights = np.array([1.0, 1.0, 1.0])
        distance = DistanceCalculator.calculate_distance(
            self.point1, self.point2, DistanceMetric.TCHEBY_WEIGHTED, equal_weights
        )
        self.assertAlmostEqual(1.0, distance, places=6)
        
    def test_tcheby_weighted_requires_weights(self):
        """Test that TCHEBY_WEIGHTED requires weights parameter."""
        with self.assertRaises(ValueError) as context:
            DistanceCalculator.calculate_distance(
                self.point1, self.point2, DistanceMetric.TCHEBY_WEIGHTED
            )
        self.assertIn("Weights required", str(context.exception))
        
    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises appropriate error."""
        with self.assertRaises(ValueError) as context:
            DistanceCalculator.calculate_distance(
                self.point1, self.point2, "INVALID_METRIC"
            )
        self.assertIn("Unknown distance metric", str(context.exception))
        
    def test_distance_calculation_consistency(self):
        """Test that distance calculations are consistent and symmetric."""
        # Test symmetry: d(a,b) = d(b,a)
        distance_ab = DistanceCalculator.calculate_distance(
            self.point1, self.point2, DistanceMetric.L2_SQUARED
        )
        distance_ba = DistanceCalculator.calculate_distance(
            self.point2, self.point1, DistanceMetric.L2_SQUARED
        )
        self.assertAlmostEqual(distance_ab, distance_ba, places=6)
        
        # Test with LINF metric
        distance_ab_linf = DistanceCalculator.calculate_distance(
            self.point1, self.point3, DistanceMetric.LINF
        )
        distance_ba_linf = DistanceCalculator.calculate_distance(
            self.point3, self.point1, DistanceMetric.LINF
        )
        self.assertAlmostEqual(distance_ab_linf, distance_ba_linf, places=6)
        
    def test_distance_calculation_with_numpy_arrays(self):
        """Test that distance calculations work with different numpy array shapes."""
        # Test with 1D arrays
        arr1d_1 = np.array([1.0, 2.0])
        arr1d_2 = np.array([3.0, 4.0])
        
        distance = DistanceCalculator.calculate_distance(
            arr1d_1, arr1d_2, DistanceMetric.L2_SQUARED
        )
        # (1-3)² + (2-4)² = 4 + 4 = 8
        self.assertAlmostEqual(8.0, distance, places=6)
        
        # Test LINF with same arrays
        distance_linf = DistanceCalculator.calculate_distance(
            arr1d_1, arr1d_2, DistanceMetric.LINF
        )
        # max(|1-3|, |2-4|) = max(2, 2) = 2
        self.assertAlmostEqual(2.0, distance_linf, places=6)
        
    def test_zero_distance(self):
        """Test that distance between identical points is zero for all metrics."""
        point = np.array([1.5, 2.3, 3.7])
        
        # L2_SQUARED
        distance_l2 = DistanceCalculator.calculate_distance(
            point, point, DistanceMetric.L2_SQUARED
        )
        self.assertAlmostEqual(0.0, distance_l2, places=6)
        
        # LINF
        distance_linf = DistanceCalculator.calculate_distance(
            point, point, DistanceMetric.LINF
        )
        self.assertAlmostEqual(0.0, distance_linf, places=6)
        
        # TCHEBY_WEIGHTED
        weights = np.array([1.0, 2.0, 0.5])
        distance_weighted = DistanceCalculator.calculate_distance(
            point, point, DistanceMetric.TCHEBY_WEIGHTED, weights
        )
        self.assertAlmostEqual(0.0, distance_weighted, places=6)


if __name__ == '__main__':
    unittest.main()