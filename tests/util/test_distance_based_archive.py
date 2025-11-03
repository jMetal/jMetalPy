import random
import unittest
from unittest.mock import Mock

import numpy as np

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import DistanceBasedArchive, distance_based_subset_selection
from jmetal.util.distance import DistanceMetric


class DistanceBasedArchiveTestCase(unittest.TestCase):

    def setUp(self):
        self.archive = DistanceBasedArchive(maximum_size=5)

    def test_should_create_archive_with_correct_maximum_size(self):
        self.assertEqual(5, self.archive.maximum_size)
        self.assertEqual(0, self.archive.size())

    def test_should_add_solution_when_archive_is_empty(self):
        solution = FloatSolution([], [], 2)
        solution.objectives = [1.0, 2.0]
        
        result = self.archive.add(solution)
        
        self.assertTrue(result)
        self.assertEqual(1, self.archive.size())

    def test_should_add_multiple_non_dominated_solutions(self):
        # Create non-dominated solutions
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 2.0 - float(i)]  # Trade-off solutions
            solutions.append(solution)
            
        for solution in solutions:
            self.archive.add(solution)
            
        self.assertEqual(3, self.archive.size())

    def test_should_not_exceed_maximum_size_with_crowding_distance_for_2_objectives(self):
        # Add 7 solutions to archive with max size 5
        for i in range(7):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 6.0 - float(i)]  # Non-dominated front
            self.archive.add(solution)
            
        self.assertEqual(5, self.archive.size())

    def test_should_handle_dominated_solutions_correctly(self):
        # Add first solution
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [1.0, 1.0]
        self.archive.add(solution1)
        
        # Add dominated solution (should be rejected)
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [2.0, 2.0]  # Dominated by solution1
        result = self.archive.add(solution2)
        
        self.assertFalse(result)
        self.assertEqual(1, self.archive.size())

    def test_should_handle_dominating_solution_correctly(self):
        # Add first solution
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [2.0, 2.0]
        self.archive.add(solution1)
        
        # Add dominating solution (should replace first)
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [1.0, 1.0]  # Dominates solution1
        result = self.archive.add(solution2)
        
        self.assertTrue(result)
        self.assertEqual(1, self.archive.size())
        # Verify the dominating solution is in the archive
        self.assertEqual([1.0, 1.0], self.archive.get(0).objectives)

    def test_should_work_with_many_objectives(self):
        # Test with 5 objectives
        archive = DistanceBasedArchive(maximum_size=3)
        
        # Add non-dominated solutions
        for i in range(5):
            solution = FloatSolution([], [], 5)
            objectives = [1.0] * 5
            objectives[i] = 0.0  # Make each solution best in one objective
            solution.objectives = objectives
            archive.add(solution)
            
        self.assertEqual(3, archive.size())  # Should be limited by max size

    def test_should_reject_duplicate_solutions(self):
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [1.0, 2.0]
        
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [1.0, 2.0]  # Same objectives
        
        result1 = self.archive.add(solution1)
        result2 = self.archive.add(solution2)
        
        self.assertTrue(result1)
        self.assertFalse(result2)  # Duplicate should be rejected
        self.assertEqual(1, self.archive.size())

    def test_should_use_custom_distance_measure(self):
        # Test using different distance metrics
        archive = DistanceBasedArchive(maximum_size=3, metric=DistanceMetric.LINF)
        
        # Add non-dominated solutions (each best in one objective)
        for i in range(5):
            solution = FloatSolution([], [], 3)
            objectives = [1.0] * 3
            objectives[i % 3] = 0.0  # Make each solution best in one objective
            solution.objectives = objectives
            archive.add(solution)
            
        self.assertEqual(3, archive.size())
        
    def test_should_support_different_distance_metrics(self):
        """Test that different distance metrics work correctly."""
        # Test L2_SQUARED (default)
        archive_l2 = DistanceBasedArchive(maximum_size=2, metric=DistanceMetric.L2_SQUARED)
        
        # Test LINF (Chebyshev distance)
        archive_linf = DistanceBasedArchive(maximum_size=2, metric=DistanceMetric.LINF)
        
        # Test TCHEBY_WEIGHTED
        weights = np.array([0.5, 0.3, 0.2])
        archive_weighted = DistanceBasedArchive(maximum_size=2, metric=DistanceMetric.TCHEBY_WEIGHTED, weights=weights)
        
        # Add same solutions to all archives
        solutions = []
        for i in range(4):
            solution = FloatSolution([], [], 3)
            solution.objectives = [i * 0.25, (3-i) * 0.25, 0.5]  # Non-dominated solutions
            solutions.append(solution)
        
        for solution in solutions:
            archive_l2.add(solution)
            archive_linf.add(solution) 
            archive_weighted.add(solution)
        
        # All archives should have exactly 2 solutions (maximum_size)
        self.assertEqual(2, archive_l2.size())
        self.assertEqual(2, archive_linf.size())
        self.assertEqual(2, archive_weighted.size())
        
    def test_should_be_thread_safe(self):
        """Test thread-safe operations on the archive."""
        import threading
        import time
        
        archive = DistanceBasedArchive(maximum_size=5)
        added_count = 0
        lock = threading.Lock()
        
        def add_solutions():
            nonlocal added_count
            for i in range(10):
                solution = FloatSolution([], [], 2)
                solution.objectives = [random.random(), random.random()]
                with lock:
                    if archive.add(solution):
                        added_count += 1
                time.sleep(0.001)  # Small delay to encourage concurrency
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_solutions)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Archive should not exceed maximum size
        self.assertLessEqual(archive.size(), 5)
        
    def test_should_handle_zero_range_objectives_robustly(self):
        """Test robust handling of objectives with zero range (constant values)."""
        archive = DistanceBasedArchive(maximum_size=2)
        
        # Create solutions where one objective is constant
        for i in range(4):
            solution = FloatSolution([], [], 3)
            solution.objectives = [i * 0.25, 1.0, (3-i) * 0.25]  # Second objective is constant
            archive.add(solution)
        
        # Should handle gracefully without division by zero
        self.assertEqual(2, archive.size())


class DistanceMetricTestCase(unittest.TestCase):
    """Test cases for the DistanceCalculator class and DistanceMetric enum."""
    
    def setUp(self):
        self.point1 = np.array([0.0, 0.0, 0.0])
        self.point2 = np.array([1.0, 1.0, 1.0])
        self.point3 = np.array([0.5, 0.2, 0.8])
        
    def test_l2_squared_distance(self):
        """Test L2 squared distance calculation."""
        from jmetal.util.distance import DistanceCalculator, DistanceMetric
        
        # Distance between (0,0,0) and (1,1,1) should be 3.0 (squared Euclidean = 1² + 1² + 1²)
        distance = DistanceCalculator.calculate_distance(self.point1, self.point2, DistanceMetric.L2_SQUARED)
        self.assertAlmostEqual(3.0, distance, places=6)
        
        # Distance from point to itself should be 0
        distance = DistanceCalculator.calculate_distance(self.point1, self.point1, DistanceMetric.L2_SQUARED)
        self.assertAlmostEqual(0.0, distance, places=6)
        
    def test_linf_distance(self):
        """Test L-infinity (Chebyshev) distance calculation."""
        from jmetal.util.distance import DistanceCalculator, DistanceMetric
        
        # L-infinity distance between (0,0,0) and (1,1,1) should be 1.0 (max difference)
        distance = DistanceCalculator.calculate_distance(self.point1, self.point2, DistanceMetric.LINF)
        self.assertAlmostEqual(1.0, distance, places=6)
        
        # Test with different point
        distance = DistanceCalculator.calculate_distance(self.point1, self.point3, DistanceMetric.LINF)
        self.assertAlmostEqual(0.8, distance, places=6)  # max(|0-0.5|, |0-0.2|, |0-0.8|) = 0.8
        
    def test_tcheby_weighted_distance(self):
        """Test weighted Chebyshev distance calculation."""
        from jmetal.util.distance import DistanceCalculator, DistanceMetric
        
        weights = np.array([2.0, 1.0, 0.5])
        
        # Weighted distance between (0,0,0) and (1,1,1) with weights [2,1,0.5]
        # max(2*|0-1|, 1*|0-1|, 0.5*|0-1|) = max(2, 1, 0.5) = 2.0
        distance = DistanceCalculator.calculate_distance(self.point1, self.point2, DistanceMetric.TCHEBY_WEIGHTED, weights)
        self.assertAlmostEqual(2.0, distance, places=6)
        
        # Test with different weights
        equal_weights = np.array([1.0, 1.0, 1.0])
        distance = DistanceCalculator.calculate_distance(self.point1, self.point2, DistanceMetric.TCHEBY_WEIGHTED, equal_weights)
        self.assertAlmostEqual(1.0, distance, places=6)
        
    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises appropriate error."""
        from jmetal.util.distance import DistanceCalculator
        
        with self.assertRaises(ValueError):
            DistanceCalculator.calculate_distance(self.point1, self.point2, "INVALID_METRIC")


class DistanceBasedSubsetSelectionTestCase(unittest.TestCase):

    def test_should_return_empty_list_for_empty_input(self):
        result = distance_based_subset_selection([], 5)
        self.assertEqual([], result)

    def test_should_return_all_solutions_when_subset_size_equals_list_size(self):
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_return_all_solutions_when_subset_size_exceeds_list_size(self):
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 5)
        self.assertEqual(3, len(result))

    def test_should_raise_error_for_invalid_subset_size(self):
        solutions = [Mock()]
        
        with self.assertRaises(ValueError):
            distance_based_subset_selection(solutions, 0)
            
        with self.assertRaises(ValueError):
            distance_based_subset_selection(solutions, -1)

    def test_should_use_crowding_distance_for_2_objectives(self):
        # Create solutions on a non-dominated front
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 4.0 - float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_use_distance_based_selection_for_many_objectives(self):
        # Test with 4 objectives
        solutions = []
        for i in range(6):
            solution = FloatSolution([], [], 4)
            solution.objectives = [float(i), float(i+1), float(i+2), float(i+3)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_handle_single_solution_selection(self):
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 3)
            solution.objectives = [float(i), float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 1)
        self.assertEqual(1, len(result))

    def test_should_handle_constant_objectives(self):
        # Test with some constant objectives (zero range)
        solutions = []
        for i in range(4):
            solution = FloatSolution([], [], 3)
            solution.objectives = [1.0, float(i), 2.0]  # First and third objectives constant
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 2)
        self.assertEqual(2, len(result))

    def test_should_be_deterministic_with_fixed_seed(self):
        # Test that results are deterministic when random seed is fixed
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 3)
            solution.objectives = [float(i), float(i+1), float(i+2)]
            solutions.append(solution)
        
        # Run twice with same seed
        random.seed(42)
        result1 = distance_based_subset_selection(solutions, 3)
        
        random.seed(42)
        result2 = distance_based_subset_selection(solutions, 3)
        
        # Results should be identical
        self.assertEqual(len(result1), len(result2))
        for sol1, sol2 in zip(result1, result2):
            self.assertEqual(sol1.objectives, sol2.objectives)


if __name__ == '__main__':
    unittest.main()