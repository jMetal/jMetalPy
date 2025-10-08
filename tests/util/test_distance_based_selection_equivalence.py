import unittest
import numpy as np
import random
from typing import List
from unittest.mock import patch

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import _robust_distance_based_selection, _original_subset_selection, _vectorized_subset_selection
from jmetal.util.distance import DistanceMetric, DistanceCalculator


class TestDistanceBasedSelectionEquivalence(unittest.TestCase):
    """
    Test suite to verify that vectorized and original implementations 
    produce exactly the same results.
    """

    def setUp(self):
        """Set up test fixtures with deterministic data."""
        # Set deterministic seed for reproducible tests
        random.seed(42)
        np.random.seed(42)
        
        # Create test solutions with known objectives
        self.test_solutions_3d = self._create_test_solutions_3d()
        self.test_solutions_2d = self._create_test_solutions_2d()
        
    def _create_test_solutions_3d(self) -> List[FloatSolution]:
        """Create a set of test solutions with 3 objectives."""
        solutions = []
        
        # Diverse set of objectives to test different scenarios
        objectives_data = [
            [0.1, 0.2, 0.3],    # Point 1
            [0.4, 0.1, 0.5],    # Point 2  
            [0.2, 0.6, 0.1],    # Point 3
            [0.8, 0.1, 0.1],    # Point 4
            [0.1, 0.8, 0.1],    # Point 5
            [0.1, 0.1, 0.8],    # Point 6
            [0.3, 0.3, 0.4],    # Point 7
            [0.5, 0.2, 0.3],    # Point 8
            [0.2, 0.5, 0.3],    # Point 9
            [0.3, 0.2, 0.5],    # Point 10
        ]
        
        for i, objectives in enumerate(objectives_data):
            solution = FloatSolution([], [], 3)
            solution.objectives = objectives
            solution.variables = [0.0] * 5  # Dummy variables
            solutions.append(solution)
            
        return solutions
    
    def _create_test_solutions_2d(self) -> List[FloatSolution]:
        """Create a set of test solutions with 2 objectives.""" 
        solutions = []
        
        objectives_data = [
            [0.1, 0.9],
            [0.2, 0.8], 
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
        
        for objectives in objectives_data:
            solution = FloatSolution([], [], 2)
            solution.objectives = objectives
            solution.variables = [0.0] * 5
            solutions.append(solution)
            
        return solutions

    def test_distance_calculator_equivalence(self):
        """Test that DistanceCalculator methods produce consistent results."""
        point1 = np.array([0.1, 0.2, 0.3])
        point2 = np.array([0.4, 0.5, 0.6])
        
        # Test L2_SQUARED metric
        individual_dist = DistanceCalculator.calculate_distance(
            point1, point2, DistanceMetric.L2_SQUARED
        )
        
        # Create matrix for vectorized calculation
        points = np.array([point1, point2])
        distance_matrix = DistanceCalculator.calculate_distance_matrix(
            points, DistanceMetric.L2_SQUARED
        )
        
        vectorized_dist = distance_matrix[0, 1]
        
        self.assertAlmostEqual(individual_dist, vectorized_dist, places=10,
                              msg="Individual and vectorized L2_SQUARED distance should be identical")

    def test_min_distances_vectorized_equivalence(self):
        """Test that vectorized min distance calculation matches iterative approach."""
        # Create test data
        points = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6], 
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7]
        ])
        
        selected_indices = [0, 2]  # Select first and third points
        
        # Calculate using vectorized method
        vectorized_min_dists = DistanceCalculator.calculate_min_distances_vectorized(
            points, selected_indices, DistanceMetric.L2_SQUARED
        )
        
        # Calculate using iterative method for comparison
        iterative_min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            min_dist = np.inf
            for selected_idx in selected_indices:
                if i != selected_idx:
                    dist = DistanceCalculator.calculate_distance(
                        points[i], points[selected_idx], DistanceMetric.L2_SQUARED
                    )
                    min_dist = min(min_dist, dist)
            iterative_min_dists[i] = min_dist
        
        # Selected points should have infinite distance
        for idx in selected_indices:
            iterative_min_dists[idx] = np.inf
            
        np.testing.assert_array_almost_equal(
            vectorized_min_dists, iterative_min_dists, decimal=10,
            err_msg="Vectorized and iterative min distance calculations should be identical"
        )

    @patch('random.randint')
    @patch('numpy.random.seed')
    def test_subset_selection_deterministic_equivalence(self, mock_np_seed, mock_randint):
        """Test that original and vectorized subset selection produce identical results with same seed."""
        # Set deterministic behavior
        mock_randint.return_value = 0  # Always select first valid dimension
        
        # Create normalized matrix from test solutions
        objectives_matrix = np.array([sol.objectives for sol in self.test_solutions_3d])
        valid_dims = [0, 1, 2]  # All dimensions valid
        
        normalized_matrix = np.zeros((len(self.test_solutions_3d), len(valid_dims)))
        min_vals = np.min(objectives_matrix[:, valid_dims], axis=0)
        max_vals = np.max(objectives_matrix[:, valid_dims], axis=0)
        ranges = max_vals - min_vals
        
        for i in range(len(self.test_solutions_3d)):
            normalized_matrix[i] = (objectives_matrix[i, valid_dims] - min_vals) / ranges
        
        # Set deterministic seed
        seed_idx = 0  # Use first solution as seed
        subset_size = 5
        
        # Test with fixed random seed to ensure deterministic behavior
        random.seed(123)
        np.random.seed(123)
        original_selection = _original_subset_selection(
            self.test_solutions_3d, normalized_matrix, subset_size, 
            seed_idx, DistanceMetric.L2_SQUARED, None
        )
        
        random.seed(123)
        np.random.seed(123)
        vectorized_selection = _vectorized_subset_selection(
            self.test_solutions_3d, normalized_matrix, subset_size,
            seed_idx, DistanceMetric.L2_SQUARED, None  
        )
        
        # Compare selected solutions by their objectives
        original_objectives = [sol.objectives for sol in original_selection]
        vectorized_objectives = [sol.objectives for sol in vectorized_selection]
        
        self.assertEqual(len(original_selection), len(vectorized_selection),
                        "Both methods should select the same number of solutions")
        
        # The selections should be identical (same solutions in same order)
        np.testing.assert_array_almost_equal(
            original_objectives, vectorized_objectives, decimal=10,
            err_msg="Original and vectorized selection should produce identical results"
        )

    def test_subset_selection_different_sizes(self):
        """Test equivalence for different subset sizes."""
        objectives_matrix = np.array([sol.objectives for sol in self.test_solutions_3d])
        valid_dims = [0, 1, 2]
        
        normalized_matrix = np.zeros((len(self.test_solutions_3d), len(valid_dims)))
        min_vals = np.min(objectives_matrix[:, valid_dims], axis=0) 
        max_vals = np.max(objectives_matrix[:, valid_dims], axis=0)
        ranges = max_vals - min_vals
        
        for i in range(len(self.test_solutions_3d)):
            normalized_matrix[i] = (objectives_matrix[i, valid_dims] - min_vals) / ranges
        
        seed_idx = 0
        
        for subset_size in [3, 5, 7, 9]:
            with self.subTest(subset_size=subset_size):
                # Set same seed for both methods
                random.seed(456)
                np.random.seed(456)
                original = _original_subset_selection(
                    self.test_solutions_3d, normalized_matrix, subset_size,
                    seed_idx, DistanceMetric.L2_SQUARED, None
                )
                
                random.seed(456)
                np.random.seed(456)
                vectorized = _vectorized_subset_selection(
                    self.test_solutions_3d, normalized_matrix, subset_size,
                    seed_idx, DistanceMetric.L2_SQUARED, None
                )
                
                original_obj = [sol.objectives for sol in original]
                vectorized_obj = [sol.objectives for sol in vectorized]
                
                np.testing.assert_array_almost_equal(
                    original_obj, vectorized_obj, decimal=10,
                    err_msg=f"Methods should be equivalent for subset_size={subset_size}"
                )

    def test_distance_metrics_equivalence(self):
        """Test equivalence across different distance metrics."""
        objectives_matrix = np.array([sol.objectives for sol in self.test_solutions_3d])
        valid_dims = [0, 1, 2]
        
        normalized_matrix = np.zeros((len(self.test_solutions_3d), len(valid_dims)))
        min_vals = np.min(objectives_matrix[:, valid_dims], axis=0)
        max_vals = np.max(objectives_matrix[:, valid_dims], axis=0)
        ranges = max_vals - min_vals
        
        for i in range(len(self.test_solutions_3d)):
            normalized_matrix[i] = (objectives_matrix[i, valid_dims] - min_vals) / ranges
        
        metrics_to_test = [
            DistanceMetric.L2_SQUARED,
            DistanceMetric.LINF,
        ]
        
        for metric in metrics_to_test:
            with self.subTest(metric=metric):
                random.seed(789)
                np.random.seed(789)
                original = _original_subset_selection(
                    self.test_solutions_3d, normalized_matrix, 5,
                    0, metric, None
                )
                
                random.seed(789)
                np.random.seed(789)
                vectorized = _vectorized_subset_selection(
                    self.test_solutions_3d, normalized_matrix, 5,
                    0, metric, None
                )
                
                original_obj = [sol.objectives for sol in original]
                vectorized_obj = [sol.objectives for sol in vectorized]
                
                np.testing.assert_array_almost_equal(
                    original_obj, vectorized_obj, decimal=10,
                    err_msg=f"Methods should be equivalent for metric={metric}"
                )

    def test_edge_cases(self):
        """Test edge cases that might reveal implementation differences."""
        
        # Test with single solution subset
        objectives_matrix = np.array([sol.objectives for sol in self.test_solutions_3d])
        valid_dims = [0, 1, 2]
        
        normalized_matrix = np.zeros((len(self.test_solutions_3d), len(valid_dims)))
        min_vals = np.min(objectives_matrix[:, valid_dims], axis=0)
        max_vals = np.max(objectives_matrix[:, valid_dims], axis=0)
        ranges = max_vals - min_vals
        
        for i in range(len(self.test_solutions_3d)):
            normalized_matrix[i] = (objectives_matrix[i, valid_dims] - min_vals) / ranges
        
        # Test subset size = 1 (only seed)
        random.seed(999)
        original = _original_subset_selection(
            self.test_solutions_3d, normalized_matrix, 1, 0, DistanceMetric.L2_SQUARED, None
        )
        
        random.seed(999)
        vectorized = _vectorized_subset_selection(
            self.test_solutions_3d, normalized_matrix, 1, 0, DistanceMetric.L2_SQUARED, None
        )
        
        self.assertEqual(len(original), 1)
        self.assertEqual(len(vectorized), 1)
        self.assertEqual(original[0].objectives, vectorized[0].objectives)


if __name__ == '__main__':
    unittest.main(verbosity=2)