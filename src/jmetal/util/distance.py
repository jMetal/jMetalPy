from abc import ABC, abstractmethod
from typing import List, Union

import numpy
from scipy.spatial import distance

"""
.. module:: distance
   :platform: Unix, Windows
   :synopsis: implementation of distances between entities

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


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
