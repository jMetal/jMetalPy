import unittest
import numpy as np

from jmetal.util.distance import EuclideanDistance, CosineDistance


class EuclideanDistanceTestCases(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.distance = EuclideanDistance()

    def test_should_get_distance_work_properly_case_1(self):
        """Case 1: [1], [1] -> distance == 0"""
        self.assertEqual(0, self.distance.get_distance([1], [1]))

    def test_should_get_distance_work_properly_case_2(self):
        """Case 2: [1, 0, 0], [0, 1, 0] -> distance == sqrt(2)"""
        expected = 1.4142135623730951
        result = self.distance.get_distance([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(expected, result, places=10)

    def test_should_get_distance_work_properly_case_3(self):
        """Case 3: [1, 1, 0], [0, 1, 0] -> distance == 1.0"""
        self.assertEqual(1.0, self.distance.get_distance([1, 1, 0], [0, 1, 0]))

    def test_should_handle_identical_points(self):
        """Test distance between identical points is zero"""
        points = [
            [0, 0],
            [1, 2, 3],
            [-1, -2, -3],
            [1.5, 2.7, 3.14, 0.0]
        ]
        for point in points:
            with self.subTest(point=point):
                self.assertEqual(0.0, self.distance.get_distance(point, point))

    def test_should_handle_numpy_arrays(self):
        """Test that numpy arrays work as input"""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        expected = np.sqrt(27.0)  # sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(9+9+9)
        result = self.distance.get_distance(arr1, arr2)
        self.assertAlmostEqual(expected, result, places=10)

    def test_should_handle_mixed_input_types(self):
        """Test mixing lists and numpy arrays"""
        list_point = [1.0, 2.0, 3.0]
        array_point = np.array([4.0, 5.0, 6.0])
        expected = np.sqrt(27.0)
        
        # List first, array second
        result1 = self.distance.get_distance(list_point, array_point)
        self.assertAlmostEqual(expected, result1, places=10)
        
        # Array first, list second  
        result2 = self.distance.get_distance(array_point, list_point)
        self.assertAlmostEqual(expected, result2, places=10)

    def test_should_handle_single_dimension(self):
        """Test 1D distance calculation"""
        result = self.distance.get_distance([5], [2])
        self.assertEqual(3.0, result)

    def test_should_handle_high_dimensions(self):
        """Test high-dimensional distance calculation"""
        # 5D points
        point1 = [1, 2, 3, 4, 5]
        point2 = [6, 7, 8, 9, 10]
        expected = np.sqrt(5 * 25)  # sqrt(5²+5²+5²+5²+5²) = sqrt(125) = 5*sqrt(5)
        result = self.distance.get_distance(point1, point2)
        self.assertAlmostEqual(expected, result, places=10)

    def test_should_handle_floating_point_precision(self):
        """Test floating point precision"""
        point1 = [0.1, 0.2, 0.3]
        point2 = [0.4, 0.5, 0.6]
        # Manually calculated: sqrt((0.3)² + (0.3)² + (0.3)²) = sqrt(0.27) = 0.3*sqrt(3)
        expected = 0.3 * np.sqrt(3)
        result = self.distance.get_distance(point1, point2)
        self.assertAlmostEqual(expected, result, places=10)

    def test_should_raise_error_for_empty_inputs(self):
        """Test that empty inputs raise ValueError"""
        with self.assertRaises(ValueError):
            self.distance.get_distance([], [1, 2])
        
        with self.assertRaises(ValueError):
            self.distance.get_distance([1, 2], [])
        
        with self.assertRaises(ValueError):
            self.distance.get_distance([], [])

    def test_should_raise_error_for_mismatched_dimensions(self):
        """Test that mismatched dimensions raise ValueError"""
        with self.assertRaises(ValueError):
            self.distance.get_distance([1, 2], [1, 2, 3])
        
        with self.assertRaises(ValueError):
            self.distance.get_distance([1], [1, 2, 3, 4])

    def test_should_raise_error_for_non_numeric_inputs(self):
        """Test that non-numeric inputs raise TypeError"""
        with self.assertRaises(TypeError):
            self.distance.get_distance(['a', 'b'], [1, 2])
        
        with self.assertRaises(TypeError):
            self.distance.get_distance([1, 2], [None, 'string'])

    def test_should_handle_negative_numbers(self):
        """Test distance calculation with negative numbers"""
        point1 = [-1, -2, -3]
        point2 = [1, 2, 3]
        expected = np.sqrt(4 + 16 + 36)  # sqrt(2² + 4² + 6²) = sqrt(56)
        result = self.distance.get_distance(point1, point2)
        self.assertAlmostEqual(expected, result, places=10)

    def test_should_handle_zero_vectors(self):
        """Test distance calculation with zero vectors"""
        zero_vector = [0, 0, 0]
        point = [3, 4, 0]
        expected = 5.0  # sqrt(3² + 4² + 0²) = 5
        result = self.distance.get_distance(zero_vector, point)
        self.assertEqual(expected, result)

    def test_should_be_symmetric(self):
        """Test that distance is symmetric: d(a,b) = d(b,a)"""
        point1 = [1, 2, 3, 4]
        point2 = [5, 6, 7, 8]
        
        result1 = self.distance.get_distance(point1, point2)
        result2 = self.distance.get_distance(point2, point1)
        
        self.assertEqual(result1, result2)

    def test_should_satisfy_triangle_inequality(self):
        """Test that triangle inequality holds: d(a,c) ≤ d(a,b) + d(b,c)"""
        point_a = [0, 0]
        point_b = [1, 1]  
        point_c = [2, 0]
        
        dist_ac = self.distance.get_distance(point_a, point_c)
        dist_ab = self.distance.get_distance(point_a, point_b)
        dist_bc = self.distance.get_distance(point_b, point_c)
        
        self.assertLessEqual(dist_ac, dist_ab + dist_bc)

    def test_performance_with_large_vectors(self):
        """Test performance with reasonably large vectors"""
        # 1000-dimensional vectors
        large_point1 = np.random.random(1000)
        large_point2 = np.random.random(1000)
        
        # Should complete without error
        result = self.distance.get_distance(large_point1, large_point2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


class CosineDistanceTestCases(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reference_point_2d = [0.0, 0.0]
        self.reference_point_3d = [1.0, 1.0, 1.0]
        
    def test_should_constructor_work_with_valid_reference_points(self):
        """Test constructor with various valid reference points"""
        # List reference point
        distance = CosineDistance([0.0, 0.0])
        self.assertIsNotNone(distance)
        
        # Numpy array reference point
        distance = CosineDistance(np.array([1.0, 2.0, 3.0]))
        self.assertIsNotNone(distance)
        
        # Single element
        distance = CosineDistance([5.0])
        self.assertIsNotNone(distance)

    def test_should_raise_error_for_invalid_reference_points(self):
        """Test that invalid reference points raise appropriate errors"""
        # Empty reference point
        with self.assertRaises(ValueError):
            CosineDistance([])
            
        # Non-numeric reference point
        with self.assertRaises(TypeError):
            CosineDistance(['a', 'b'])
            
        # None reference point
        with self.assertRaises(TypeError):
            CosineDistance(None)

    def test_should_identical_points_have_distance_of_zero(self):
        """Test that identical points have distance of zero"""
        distance = CosineDistance(self.reference_point_2d)
        
        test_points = [
            [1.0, 1.0],
            [2.0, 2.0], 
            [-1.0, -1.0],
            [0.5, -0.5]
        ]
        
        for point in test_points:
            with self.subTest(point=point):
                result = distance.get_distance(point, point)
                self.assertEqual(0.0, result)

    def test_should_points_in_same_direction_have_distance_of_zero(self):
        """Test that points in the same direction from reference have distance 0"""
        distance = CosineDistance(self.reference_point_2d)
        
        # Points [1,1] and [2,2] are in same direction from [0,0]
        result = distance.get_distance([1.0, 1.0], [2.0, 2.0])
        self.assertAlmostEqual(0.0, result, places=10)
        
        # Points [3,3] and [0.5,0.5] are in same direction from [0,0]  
        result = distance.get_distance([3.0, 3.0], [0.5, 0.5])
        self.assertAlmostEqual(0.0, result, places=10)

    def test_should_orthogonal_points_have_distance_of_one(self):
        """Test that orthogonal points have distance of 1"""
        distance = CosineDistance(self.reference_point_2d)
        
        # [0,1] and [1,0] are orthogonal from [0,0]
        result = distance.get_distance([0.0, 1.0], [1.0, 0.0])
        self.assertAlmostEqual(1.0, result, places=10)
        
        # [2,0] and [0,3] are orthogonal from [0,0]
        result = distance.get_distance([2.0, 0.0], [0.0, 3.0])
        self.assertAlmostEqual(1.0, result, places=10)

    def test_should_opposite_points_have_distance_of_two(self):
        """Test that opposite points have distance of 2"""
        distance = CosineDistance(self.reference_point_2d)
        
        # [1,0] and [-1,0] are opposite from [0,0]
        result = distance.get_distance([1.0, 0.0], [-1.0, 0.0])
        self.assertAlmostEqual(2.0, result, places=10)
        
        # [2,2] and [-1,-1] are opposite from [0,0]  
        result = distance.get_distance([2.0, 2.0], [-1.0, -1.0])
        self.assertAlmostEqual(2.0, result, places=10)

    def test_should_handle_reference_point_translation(self):
        """Test distance calculation with non-zero reference point"""
        reference = [1.0, 1.0]
        distance = CosineDistance(reference)
        
        # Points [2,2] and [3,3] become [1,1] and [2,2] after translation
        # These are in same direction, so distance should be 0
        result = distance.get_distance([2.0, 2.0], [3.0, 3.0])
        self.assertAlmostEqual(0.0, result, places=10)
        
        # Points [2,1] and [1,2] become [1,0] and [0,1] after translation
        # These are orthogonal, so distance should be 1
        result = distance.get_distance([2.0, 1.0], [1.0, 2.0])
        self.assertAlmostEqual(1.0, result, places=10)

    def test_should_handle_points_at_reference_point(self):
        """Test handling when one or both points are at reference point"""
        distance = CosineDistance([1.0, 1.0])
        
        # Both points at reference point
        result = distance.get_distance([1.0, 1.0], [1.0, 1.0])
        self.assertEqual(0.0, result)
        
        # One point at reference point
        result = distance.get_distance([1.0, 1.0], [2.0, 2.0])
        self.assertEqual(1.0, result)
        
        result = distance.get_distance([2.0, 2.0], [1.0, 1.0])
        self.assertEqual(1.0, result)

    def test_should_handle_numpy_arrays(self):
        """Test that numpy arrays work as input"""
        distance = CosineDistance(np.array([0.0, 0.0]))
        
        arr1 = np.array([1.0, 0.0])
        arr2 = np.array([0.0, 1.0])
        
        # These are orthogonal
        result = distance.get_distance(arr1, arr2)
        self.assertAlmostEqual(1.0, result, places=10)

    def test_should_handle_mixed_input_types(self):
        """Test mixing lists and numpy arrays"""
        distance = CosineDistance([0.0, 0.0])
        
        list_point = [1.0, 1.0]
        array_point = np.array([2.0, 2.0])
        
        # Same direction, should be 0
        result = distance.get_distance(list_point, array_point)
        self.assertAlmostEqual(0.0, result, places=10)

    def test_should_handle_high_dimensions(self):
        """Test high-dimensional cosine distance"""
        # 5D reference point
        ref_point = [0.0, 0.0, 0.0, 0.0, 0.0]
        distance = CosineDistance(ref_point)
        
        # Same direction vectors
        point1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        point2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # 2 * point1
        
        result = distance.get_distance(point1, point2)
        self.assertAlmostEqual(0.0, result, places=10)

    def test_should_validate_input_dimensions(self):
        """Test that dimension mismatches raise errors"""
        distance = CosineDistance([0.0, 0.0])  # 2D reference
        
        # Different dimensions between inputs
        with self.assertRaises(ValueError):
            distance.get_distance([1.0, 2.0], [1.0, 2.0, 3.0])
        
        # Wrong dimension vs reference point
        with self.assertRaises(ValueError):
            distance.get_distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    def test_should_validate_empty_inputs(self):
        """Test that empty inputs raise errors"""
        distance = CosineDistance([1.0, 2.0])
        
        with self.assertRaises(ValueError):
            distance.get_distance([], [1.0, 2.0])
            
        with self.assertRaises(ValueError):
            distance.get_distance([1.0, 2.0], [])

    def test_should_validate_non_numeric_inputs(self):
        """Test that non-numeric inputs raise errors"""
        distance = CosineDistance([0.0, 0.0])
        
        with self.assertRaises(TypeError):
            distance.get_distance(['a', 'b'], [1.0, 2.0])
            
        with self.assertRaises(TypeError):
            distance.get_distance([1.0, 2.0], [None, 'test'])

    def test_should_distance_be_in_valid_range(self):
        """Test that distance is always in range [0, 2]"""
        distance = CosineDistance([0.0, 0.0])
        
        test_cases = [
            ([1.0, 0.0], [0.0, 1.0]),  # Orthogonal
            ([1.0, 1.0], [2.0, 2.0]),  # Same direction
            ([1.0, 0.0], [-1.0, 0.0]), # Opposite
            ([3.0, 4.0], [-3.0, -4.0]), # Opposite scaled
            ([1.0, 2.0], [2.0, 1.0]),  # Different
        ]
        
        for point1, point2 in test_cases:
            with self.subTest(point1=point1, point2=point2):
                result = distance.get_distance(point1, point2)
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 2.0)

    def test_should_be_symmetric(self):
        """Test that distance is symmetric: d(a,b) = d(b,a)"""
        distance = CosineDistance([1.0, 1.0])
        
        point1 = [2.0, 3.0]
        point2 = [4.0, 1.0]
        
        result1 = distance.get_distance(point1, point2)
        result2 = distance.get_distance(point2, point1)
        
        self.assertAlmostEqual(result1, result2, places=10)

    def test_get_similarity_method(self):
        """Test the get_similarity convenience method"""
        distance = CosineDistance([0.0, 0.0])
        
        # Same direction should have similarity = 1
        similarity = distance.get_similarity([1.0, 1.0], [2.0, 2.0])
        self.assertAlmostEqual(1.0, similarity, places=10)
        
        # Orthogonal should have similarity = 0
        similarity = distance.get_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(0.0, similarity, places=10)
        
        # Opposite should have similarity = -1
        similarity = distance.get_similarity([1.0, 0.0], [-1.0, 0.0])
        self.assertAlmostEqual(-1.0, similarity, places=10)

    def test_numerical_stability(self):
        """Test numerical stability with very small and large values"""
        distance = CosineDistance([0.0, 0.0])
        
        # Very small values
        result = distance.get_distance([1e-10, 1e-10], [2e-10, 2e-10])
        self.assertAlmostEqual(0.0, result, places=8)
        
        # Very large values
        result = distance.get_distance([1e10, 0.0], [0.0, 1e10])
        self.assertAlmostEqual(1.0, result, places=8)

if __name__ == "__main__":
    unittest.main()
