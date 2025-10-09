from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Optional

import numpy
from scipy.spatial import distance

"""
.. module:: distance
   :platform: Unix, Windows
   :synopsis: implementation of distances between entities

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class DistanceMetric(Enum):
    """
    Enumeration of available distance metrics for optimized distance calculations.
    
    Each metric is optimized for specific use cases:
    - L2_SQUARED: Fastest, avoids sqrt computation for relative distance comparisons
    - LINF: Efficient for high-dimensional spaces, emphasizes maximum difference
    - TCHEBY_WEIGHTED: Flexible weighted distance allowing preference specification
    """
    L2_SQUARED = "l2_squared"
    LINF = "linf"  
    TCHEBY_WEIGHTED = "tcheby_weighted"


class DistanceCalculator:
    """
    High-performance distance calculator supporting multiple metrics.
    
    This utility class provides optimized implementations for different distance
    metrics commonly used in multi-objective optimization. All methods are static
    for efficiency and support numpy array operations for vectorized performance.
    """
    
    @staticmethod
    def calculate_distance(point1: numpy.ndarray, point2: numpy.ndarray, 
                          metric: DistanceMetric, weights: Optional[numpy.ndarray] = None) -> float:
        """
        Calculate distance between two points using the specified metric.
        
        Args:
            point1: First point as numpy array
            point2: Second point as numpy array  
            metric: Distance metric to use
            weights: Optional weights for TCHEBY_WEIGHTED metric
            
        Returns:
            float: Distance between points according to specified metric
            
        Raises:
            ValueError: If metric is invalid or weights are required but not provided
        """
        if metric == DistanceMetric.L2_SQUARED:
            return DistanceCalculator._l2_squared(point1, point2)
        elif metric == DistanceMetric.LINF:
            return DistanceCalculator._linf(point1, point2)
        elif metric == DistanceMetric.TCHEBY_WEIGHTED:
            if weights is None:
                raise ValueError("Weights required for TCHEBY_WEIGHTED metric")
            return DistanceCalculator._tcheby_weighted(point1, point2, weights)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    @staticmethod
    def _l2_squared(point1: numpy.ndarray, point2: numpy.ndarray) -> float:
        """
        Calculate squared Euclidean distance (L2 squared).
        
        This is faster than regular Euclidean distance as it avoids the sqrt operation.
        Suitable for relative distance comparisons where absolute values don't matter.
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            float: Squared Euclidean distance
        """
        diff = point1 - point2
        return numpy.dot(diff, diff)
    
    @staticmethod
    def _linf(point1: numpy.ndarray, point2: numpy.ndarray) -> float:
        """
        Calculate L-infinity (Chebyshev) distance.
        
        This metric emphasizes the maximum difference across all dimensions.
        Efficient for high-dimensional spaces and robust to outliers.
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            float: L-infinity distance (maximum absolute difference)
        """
        return numpy.max(numpy.abs(point1 - point2))
    
    @staticmethod
    def _tcheby_weighted(point1: numpy.ndarray, point2: numpy.ndarray, 
                        weights: numpy.ndarray) -> float:
        """
        Calculate weighted Chebyshev distance.
        
        This metric allows specifying importance weights for different dimensions.
        Useful when objectives have different scales or importance levels.
        
        Args:
            point1: First point
            point2: Second point
            weights: Weight vector for each dimension
            
        Returns:
            float: Weighted Chebyshev distance
        """
        weighted_diff = weights * numpy.abs(point1 - point2)
        return numpy.max(weighted_diff)
    
    @staticmethod
    def calculate_distance_matrix(points: numpy.ndarray, metric: DistanceMetric,
                                 weights: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        """
        Calculate pairwise distance matrix between all points using vectorized operations.
        
        This is significantly faster than calculating distances individually,
        especially for large numbers of points. Uses optimized NumPy operations
        for maximum performance.
        
        Args:
            points: 2D array where each row is a point (n_points, n_dimensions)
            metric: Distance metric to use
            weights: Optional weights for TCHEBY_WEIGHTED metric
            
        Returns:
            numpy.ndarray: Symmetric distance matrix (n_points, n_points)
            
        Raises:
            ValueError: If metric is invalid or weights are required but not provided
        """
        n_points = points.shape[0]
        
        if metric == DistanceMetric.L2_SQUARED:
            return DistanceCalculator._l2_squared_matrix(points)
        elif metric == DistanceMetric.LINF:
            return DistanceCalculator._linf_matrix(points)
        elif metric == DistanceMetric.TCHEBY_WEIGHTED:
            if weights is None:
                raise ValueError("Weights required for TCHEBY_WEIGHTED metric")
            return DistanceCalculator._tcheby_weighted_matrix(points, weights)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    @staticmethod
    def _l2_squared_matrix(points: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate squared Euclidean distance matrix using vectorized operations.
        
        Uses broadcasting and matrix operations for optimal performance.
        Formula: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
        
        Args:
            points: 2D array of points (n_points, n_dimensions)
            
        Returns:
            numpy.ndarray: Symmetric squared distance matrix
        """
        # Calculate squared norms for each point
        squared_norms = numpy.sum(points**2, axis=1)
        
        # Calculate dot products matrix
        dot_products = numpy.dot(points, points.T)
        
        # Use broadcasting to create distance matrix
        # ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
        distance_matrix = (squared_norms[:, numpy.newaxis] + 
                          squared_norms[numpy.newaxis, :] - 
                          2 * dot_products)
        
        # Ensure diagonal is exactly zero (numerical precision)
        numpy.fill_diagonal(distance_matrix, 0.0)
        
        # Ensure non-negative distances (numerical precision)
        distance_matrix = numpy.maximum(distance_matrix, 0.0)
        
        return distance_matrix
    
    @staticmethod
    def _linf_matrix(points: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate L-infinity distance matrix using vectorized operations.
        
        Uses broadcasting for efficient computation of maximum absolute differences.
        
        Args:
            points: 2D array of points (n_points, n_dimensions)
            
        Returns:
            numpy.ndarray: Symmetric L-infinity distance matrix
        """
        n_points = points.shape[0]
        
        # Reshape for broadcasting: (n_points, 1, n_dims) - (1, n_points, n_dims)
        points_expanded1 = points[:, numpy.newaxis, :]
        points_expanded2 = points[numpy.newaxis, :, :]
        
        # Calculate absolute differences and take maximum along last axis
        distance_matrix = numpy.max(numpy.abs(points_expanded1 - points_expanded2), axis=2)
        
        return distance_matrix
    
    @staticmethod
    def _tcheby_weighted_matrix(points: numpy.ndarray, weights: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate weighted Chebyshev distance matrix using vectorized operations.
        
        Args:
            points: 2D array of points (n_points, n_dimensions)
            weights: Weight vector for each dimension
            
        Returns:
            numpy.ndarray: Symmetric weighted Chebyshev distance matrix
        """
        n_points = points.shape[0]
        
        # Reshape for broadcasting
        points_expanded1 = points[:, numpy.newaxis, :]
        points_expanded2 = points[numpy.newaxis, :, :]
        
        # Apply weights and calculate maximum weighted difference
        weighted_diff = weights * numpy.abs(points_expanded1 - points_expanded2)
        distance_matrix = numpy.max(weighted_diff, axis=2)
        
        return distance_matrix
    
    @staticmethod
    def calculate_min_distances_vectorized(points: numpy.ndarray, 
                                         selected_indices: List[int],
                                         metric: DistanceMetric,
                                         weights: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        """
        Calculate minimum distances from each point to the selected points using vectorized operations.
        
        This is optimized for the subset selection process where we need to find
        the minimum distance from each candidate point to any already selected point.
        
        Args:
            points: 2D array of all points (n_points, n_dimensions)
            selected_indices: List of indices of already selected points
            metric: Distance metric to use
            weights: Optional weights for TCHEBY_WEIGHTED metric
            
        Returns:
            numpy.ndarray: Array of minimum distances for each point
        """
        if len(selected_indices) == 0:
            # If no points selected yet, return infinite distances
            return numpy.full(points.shape[0], numpy.inf)
        
        # Extract selected points
        selected_points = points[selected_indices]
        
        # Calculate minimum distances
        if metric == DistanceMetric.L2_SQUARED:
            min_distances = DistanceCalculator._min_l2_squared_distances(points, selected_points)
        elif metric == DistanceMetric.LINF:
            min_distances = DistanceCalculator._min_linf_distances(points, selected_points)
        elif metric == DistanceMetric.TCHEBY_WEIGHTED:
            if weights is None:
                raise ValueError("Weights required for TCHEBY_WEIGHTED metric")
            min_distances = DistanceCalculator._min_tcheby_weighted_distances(points, selected_points, weights)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        
        # Set selected points to have infinite distance (they should not be reselected)
        for idx in selected_indices:
            min_distances[idx] = numpy.inf
            
        return min_distances
    
    @staticmethod
    def _min_l2_squared_distances(points: numpy.ndarray, selected_points: numpy.ndarray) -> numpy.ndarray:
        """Calculate minimum squared L2 distances to selected points."""
        # Calculate distances from each point to each selected point
        # points: (n_points, n_dims), selected_points: (n_selected, n_dims)
        
        # Squared norms
        points_norms = numpy.sum(points**2, axis=1)  # (n_points,)
        selected_norms = numpy.sum(selected_points**2, axis=1)  # (n_selected,)
        
        # Dot products matrix
        dot_products = numpy.dot(points, selected_points.T)  # (n_points, n_selected)
        
        # Distance matrix using broadcasting
        distances = (points_norms[:, numpy.newaxis] + 
                    selected_norms[numpy.newaxis, :] - 
                    2 * dot_products)
        
        # Ensure non-negative
        distances = numpy.maximum(distances, 0.0)
        
        # Return minimum distance for each point
        return numpy.min(distances, axis=1)
    
    @staticmethod
    def _min_linf_distances(points: numpy.ndarray, selected_points: numpy.ndarray) -> numpy.ndarray:
        """Calculate minimum L-infinity distances to selected points."""
        # points: (n_points, n_dims), selected_points: (n_selected, n_dims)
        
        # Reshape for broadcasting
        points_expanded = points[:, numpy.newaxis, :]  # (n_points, 1, n_dims)
        selected_expanded = selected_points[numpy.newaxis, :, :]  # (1, n_selected, n_dims)
        
        # Calculate L-infinity distances
        distances = numpy.max(numpy.abs(points_expanded - selected_expanded), axis=2)  # (n_points, n_selected)
        
        # Return minimum distance for each point
        return numpy.min(distances, axis=1)
    
    @staticmethod
    def _min_tcheby_weighted_distances(points: numpy.ndarray, 
                                     selected_points: numpy.ndarray,
                                     weights: numpy.ndarray) -> numpy.ndarray:
        """Calculate minimum weighted Chebyshev distances to selected points."""
        # Reshape for broadcasting
        points_expanded = points[:, numpy.newaxis, :]  # (n_points, 1, n_dims)
        selected_expanded = selected_points[numpy.newaxis, :, :]  # (1, n_selected, n_dims)
        
        # Apply weights and calculate weighted Chebyshev distances
        weighted_diff = weights * numpy.abs(points_expanded - selected_expanded)
        distances = numpy.max(weighted_diff, axis=2)  # (n_points, n_selected)
        
        # Return minimum distance for each point
        return numpy.min(distances, axis=1)


class Distance(ABC):
    @abstractmethod
    def get_distance(self, element1, element2) -> float:
        pass


class EuclideanDistance(Distance):
    """
    Euclidean distance implementation with enhanced type safety and validation.
    
    Supports both Python lists and numpy arrays as input.
    Provides comprehensive input validation for robust operation.
    """
    
    def get_distance(self, list1: Union[List[float], numpy.ndarray], 
                     list2: Union[List[float], numpy.ndarray]) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            list1: First point as list or numpy array
            list2: Second point as list or numpy array
            
        Returns:
            float: Euclidean distance between the points
            
        Raises:
            ValueError: If inputs have different dimensions or are empty
            TypeError: If inputs cannot be converted to numeric arrays
        """
        # Convert to numpy arrays for consistent handling
        try:
            arr1 = numpy.asarray(list1, dtype=float)
            arr2 = numpy.asarray(list2, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Input vectors must be numeric: {e}")
        
        # Input validation - check for empty arrays
        if arr1.size == 0 or arr2.size == 0:
            raise ValueError("Input vectors cannot be empty")
            
        # Dimension validation
        if arr1.shape != arr2.shape:
            raise ValueError(f"Input vectors must have the same dimensions: "
                           f"{arr1.shape} vs {arr2.shape}")
        
        # For 1D vectors, use scipy's optimized implementation
        if arr1.ndim == 1:
            return distance.euclidean(arr1, arr2)
        else:
            # For higher dimensions, flatten and compute
            return distance.euclidean(arr1.flatten(), arr2.flatten())


class CosineDistance(Distance):
    """
    Cosine distance implementation with reference point translation.
    
    This class computes the cosine distance between two points after translating them
    relative to a reference point. The cosine distance is defined as:
    distance = 1 - cosine_similarity
    
    Where cosine_similarity = (a·b) / (||a|| * ||b||)
    
    The distance ranges from 0 (identical direction) to 2 (opposite directions).
    
    Performance optimizations:
    - Cached reference point norm for efficiency
    - Vectorized numpy operations 
    - Robust input validation and error handling
    """
    
    def __init__(self, reference_point: Union[List[float], numpy.ndarray]):
        """
        Initialize the cosine distance calculator with a reference point.
        
        Args:
            reference_point: Point used to translate input vectors before computing distance
            
        Raises:
            ValueError: If reference point is empty
            TypeError: If reference point is not numeric or is None
        """
        if reference_point is None:
            raise TypeError("Reference point cannot be None")
            
        try:
            self.reference_point = numpy.asarray(reference_point, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Reference point must be numeric: {e}")
            
        if self.reference_point.size == 0:
            raise ValueError("Reference point cannot be empty")
            
        # Cache reference point norm for efficiency
        self._ref_norm = numpy.linalg.norm(self.reference_point)

    def get_distance(self, list1: Union[List[float], numpy.ndarray], 
                     list2: Union[List[float], numpy.ndarray]) -> float:
        """
        Calculate the cosine distance between two points relative to the reference point.
        
        The computation follows these steps:
        1. Translate both points by subtracting the reference point
        2. Compute cosine similarity of the translated vectors
        3. Return cosine distance = 1 - cosine_similarity
        
        Args:
            list1: First point as list or numpy array
            list2: Second point as list or numpy array
            
        Returns:
            float: Cosine distance in range [0, 2]
                  0 = vectors point in same direction
                  1 = vectors are orthogonal  
                  2 = vectors point in opposite directions
                  
        Raises:
            ValueError: If inputs have different dimensions than reference point or are empty
            TypeError: If inputs cannot be converted to numeric arrays
        """
        # Convert inputs to numpy arrays
        try:
            vec1 = numpy.asarray(list1, dtype=float)
            vec2 = numpy.asarray(list2, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Input vectors must be numeric: {e}")
        
        # Validate inputs
        if vec1.size == 0 or vec2.size == 0:
            raise ValueError("Input vectors cannot be empty")
            
        if vec1.shape != vec2.shape:
            raise ValueError(f"Input vectors must have same dimensions: {vec1.shape} vs {vec2.shape}")
            
        if vec1.shape != self.reference_point.shape:
            raise ValueError(f"Input vectors must match reference point dimensions: "
                           f"{vec1.shape} vs {self.reference_point.shape}")
        
        # Translate vectors relative to reference point
        diff1 = vec1 - self.reference_point
        diff2 = vec2 - self.reference_point
        
        # Handle zero vectors after translation (return 0 since they're at reference point)
        norm1 = numpy.linalg.norm(diff1)
        norm2 = numpy.linalg.norm(diff2)
        
        if norm1 == 0.0 and norm2 == 0.0:
            # Both vectors are at the reference point
            return 0.0
        elif norm1 == 0.0 or norm2 == 0.0:
            # One vector is at reference point, other is not
            return 1.0
        
        # Compute cosine similarity
        dot_product = numpy.dot(diff1, diff2)
        cosine_similarity = dot_product / (norm1 * norm2)
        
        # Clamp to handle numerical precision issues
        cosine_similarity = numpy.clip(cosine_similarity, -1.0, 1.0)
        
        # Return cosine distance = 1 - cosine_similarity
        distance_result = 1.0 - cosine_similarity
        
        # Handle numerical precision for identical vectors (should be exactly 0)
        if numpy.allclose(diff1, diff2, rtol=1e-15, atol=1e-15):
            return 0.0
            
        return distance_result

    def get_similarity(self, list1: Union[List[float], numpy.ndarray], 
                       list2: Union[List[float], numpy.ndarray]) -> float:
        """
        Calculate the cosine similarity between two points relative to the reference point.
        
        This is a convenience method that returns the similarity instead of distance.
        
        Args:
            list1: First point as list or numpy array
            list2: Second point as list or numpy array
            
        Returns:
            float: Cosine similarity in range [-1, 1]
                  1 = vectors point in same direction
                  0 = vectors are orthogonal
                 -1 = vectors point in opposite directions
        """
        distance = self.get_distance(list1, list2)
        return 1.0 - distance
